"""
"""
import logging
# import warnings

import numpy as np
import scipy as sp

from . import radiation, utils
from . constants import MELC, MPRT, SPLC, K_BLTZ, H_PLNK


class Mahadevan96:

    def __init__(self, adaf, freqs, log=30, backup_temp=None, quiet=True):
        """
        """
        if not isinstance(log, logging.Logger):
            log = utils.get_log(level=log)

        self.freqs = freqs
        self._log = log
        self._quiet = quiet

        # Mass in units of solar=masses
        self.msol = adaf.ms
        self.fedd = adaf.fedd

        # self._adaf = adaf
        self._alpha = adaf.alpha_visc
        self._beta = adaf.beta_gp
        self._eps_prime = adaf._eps_prime
        self._delta = MELC/MPRT  # fraction of energy transfered to electrons from viscous heating
        self._c1 = adaf._c1
        self._c3 = adaf._c3
        self._rmin = adaf.rs[0]
        self._rmax = adaf.rs[-1]

        self._s1 = 1.42e9 * np.sqrt(1 - self._beta) * np.sqrt(self._c3 / self._c1 / self._alpha)
        # s2 = 1.19e-13 * xm_char
        self._s3 = 1.05e-24

        self._pars = ["alpha", "beta", "eps_prime", "delta", "c1", "c3", "rmin", "rmax"]

        self._backup_temp = backup_temp

        # Find the electron temperature and calculate spectra
        self._solve()
        return

    def __str__(self):
        rv = "Mahadevan96(msol={:.4e}, fedd={:.4e}".format(self.msol, self.fedd)
        for pp in self._pars:
            vv = getattr(self, "_" + pp)
            rv += ", {}={:.3e}".format(pp, vv)

        rv += ")"
        return rv

    def _solve(self):
        log = self._log

        def _func(logt):
            tt = np.power(10.0, logt)
            qv, qs, qb, qc = self._heat_cool(tt)
            rv = qv - (qs + qb + qc)
            return rv

        start_temps = [1e11, 1e10, 1e12, 1e9, 1e8]
        success = False
        for ii, t0 in enumerate(start_temps):
            log.debug("Try {}, temp: {:.1e}".format(ii, t0))
            try:
                logt = sp.optimize.newton(_func, np.log10(t0), tol=1e-4, maxiter=100)
                self.temp_e = np.power(10.0, logt)
            except (RuntimeError, FloatingPointError) as err:
                log.debug("WARNING: Trial '{}' (t={:.1e}) optimization failed: {}".format(
                    ii, t0, str(err)))
            else:
                success = True
                break

        if success:
            log.debug("Success with `t0`={:.2e} ==> t={:.2e}".format(t0, self.temp_e))
        else:
            lvl = log.DEBUG if self._quiet else log.ERROR
            log.log(lvl, "FAILED to find electron temperature!")
            log.log(lvl, str(self))
            log.log(lvl, "m = {:.2e}, f = {:.2e}".format(self.msol, self.fedd))
            err = ("Unable to find electron temperature!"
                   "\nIf the eddington factor is larger than 1e-2, "
                   "this may be expected!")
            if self._backup_temp is None:
                raise RuntimeError(err)

            self.temp_e = self._backup_temp
            log.log(lvl, "WARNING: setting temperature to '{}'!".format(self.temp_e))

        qv, qs, qb, qc = self._heat_cool(self.temp_e)
        heat = qv
        cool = qs + qb + qc
        diff = np.fabs(heat - cool) / heat

        if diff > 1e-2:
            lvl = logging.DEBUG

            if not self._quiet:
                if diff > 1.0:
                    lvl = logging.ERROR
                elif diff > 1e-1:
                    lvl = logging.INFO

            err = "Electron temperature seems inconsistent (Te = {:.2e})!".format(self.temp_e)
            err += "\n\tm: {:.2e}, f: {:.2e}".format(self.msol, self.fedd)
            err += "\n\tHeating: {:.2e}, Cooling: {:.2e}, diff: {:.4e}".format(heat, cool, diff)
            err += "\n\tThis may mean there is an input error (e.g. mdot may be too large... or small?)."
            log.log(lvl, err)

        self.theta_e = radiation.dimensionless_temperature_theta(self.temp_e, MELC)
        # print("Electron effective temperature: {:.2e} K (theta = {:.2e})".format(
        #     self.temp_e, self.theta_e))
        self._xm_e = xm_from_te(self.temp_e, self.msol, self.fedd)
        self._s2 = self._const_s2(self._xm_e)

        freqs = self.freqs
        synch = self._calc_spectrum_synch(freqs)
        brems = self._calc_spectrum_brems(freqs)
        compt = self._calc_spectrum_compt(freqs)

        self.spectrum_synch = synch
        self.spectrum_brems = brems
        self.spectrum_compt = compt
        self.spectrum = synch + brems + compt
        return

    def _const_s2(self, xm):
        s2 = 1.19e-13 * xm
        return s2

    def _heat_cool(self, temp):
        """Calculate heating and cooling rates for disk as a whole.
        """

        alpha = self._alpha
        beta = self._beta
        eps_prime = self._eps_prime
        msol = self.msol
        fedd = self.fedd
        delta = self._delta
        c1 = self._c1
        c3 = self._c3
        rmin = self._rmin
        rmax = self._rmax

        theta = K_BLTZ * temp / (MELC * SPLC * SPLC)
        xm = xm_from_te(temp, msol, fedd)

        s1 = 1.42e9 * np.sqrt(1 - beta) * np.sqrt(c3 / c1 / alpha)
        s2 = self._const_s2(xm)
        s3 = 1.05e-24

        alpha_crit, mean_amp_a, tau_es = self._compton_params(temp, fedd)

        # Viscous Heating
        # ---------------
        _ge = radiation._heat_func_g(theta)
        q1 = 1.2e38 * _ge * c3 * beta * msol * np.square(fedd) / np.square(alpha*c1) / rmin
        q2 = delta * 9.39e38 * eps_prime * c3 * msol * fedd / rmin
        heat_elc = q1 + q2

        # Synchrotron
        # -----------
        # Eq. 24  [Hz]
        f_p = self._freq_synch_peak(temp, msol, fedd, s2=s2)
        lum_synch_peak = np.power(s1 * s2, 3) * s3 * np.power(rmin, -1.75) * np.sqrt(msol)
        lum_synch_peak *= np.power(fedd, 1.5) * np.power(temp, 7) / f_p

        # Eq. 26
        power_synch = 5.3e35 * np.power(xm/1000, 3) * np.power(alpha/0.3, -1.5)
        power_synch *= np.power((1 - beta)/0.5, 1.5) * np.power(c1/0.5, -1.5)

        # Bremsstrahlung
        # --------------
        # Eq. 29
        power_brems = 4.78e34 * np.log(rmax/rmin) / np.square(alpha * c1)
        power_brems *= radiation._brems_fit_func_f(theta) * fedd * msol

        # Compton
        # -------
        power_compt = lum_synch_peak * f_p / (1 - alpha_crit)
        power_compt *= (np.power(6.2e7 * (temp/1e9) / (f_p/1e12), 1 - alpha_crit) - 1.0)

        return heat_elc, power_synch, power_brems, power_compt

    def _freq_synch_peak(self, temp, msol, fedd, s2=None):
        """Mahadevan 1996 Eq. 24
        """
        if s2 is None:
            xm = xm_from_te(temp, msol, fedd)
            s2 = self._const_s2(xm)
        nu_p = self._s1 * s2 * np.sqrt(fedd/msol) * np.square(temp) * np.power(self._rmin, -1.25)
        return nu_p

    def _compton_params(self, te, fedd):
        """Mahadevan Eqs. 31-34
        """
        theta_e = radiation.dimensionless_temperature_theta(te, MELC)
        # Eq. 31
        tau_es = 23.87 * fedd * (0.3 / self._alpha) * (0.5 / self._c1) * np.sqrt(3/self._rmin)
        # Eq. 32
        mean_amp_a = 1.0 + 4.0 * theta_e + 16*np.square(theta_e)
        # Eq. 34
        alpha_crit = - np.log(tau_es) / np.log(mean_amp_a)
        return alpha_crit, mean_amp_a, tau_es

    def _synch_peak(self, fedd, msol, temp, s2=None):
        if s2 is None:
            xm = xm_from_te(temp, msol, fedd)
            s2 = self._const_s2(xm)
        f_p = self._freq_synch_peak(temp, msol, fedd, s2=s2)
        l_p = np.power(self._s1 * s2, 3) * self._s3 * np.power(self._rmin, -1.75) * np.sqrt(msol)
        l_p *= np.power(fedd, 1.5) * np.power(temp, 7) / f_p
        return f_p, l_p

    def _calc_spectrum_synch(self, freqs):
        """Mahadevan 1996 - Eq. 25

        Cutoff above peak frequency (i.e. ignore exponential portion).
        Ignore low-frequency transition to steeper (22/13 slope) from rmax.
        """
        msol = self.msol
        fedd = self.fedd

        scalar = np.isscalar(freqs)
        freqs = np.atleast_1d(freqs)

        lnu = self._s3 * np.power(self._s1*self._s2, 1.6)
        lnu *= np.power(msol, 1.2) * np.power(fedd, 0.8)
        lnu *= np.power(self.temp_e, 4.2) * np.power(freqs, 0.4)

        nu_p = self._freq_synch_peak(self.temp_e, msol, fedd, s2=self._s2)
        lnu[freqs > nu_p] = 0.0
        if scalar:
            lnu = np.squeeze(lnu)

        return lnu

    def _calc_spectrum_brems(self, freqs):
        """Mahadevan 1996 - Eq. 30
        """
        msol = self.msol
        fedd = self.fedd
        temp = self.temp_e
        const = 2.29e24   # erg/s/Hz

        scalar = np.isscalar(freqs)
        freqs = np.atleast_1d(freqs)

        t1 = np.log(self._rmax/self._rmin) / np.square(self._alpha * self._c1)
        t2 = np.exp(-H_PLNK*freqs / (K_BLTZ * temp)) * msol * np.square(fedd) / temp
        fe = radiation._brems_fit_func_f(temp)
        lbrems = const * t1 * fe * t2
        if scalar:
            lbrems = np.squeeze(lbrems)

        return lbrems

    def _calc_spectrum_compt(self, freqs):
        """Compton Scattering spectrum from upscattering of Synchrotron photons.

        Mahadevan 1996 - Eq. 38
        """
        fedd = self.fedd
        temp = self.temp_e

        scalar = np.isscalar(freqs)
        freqs = np.atleast_1d(freqs)

        f_p, l_p = self._synch_peak(fedd, self.msol, temp)
        alpha_c, mean_amp_a, tau_es = self._compton_params(temp, fedd)
        lsp = np.power(freqs/f_p, -alpha_c) * l_p
        lsp[freqs < f_p] = 0.0

        # See Eq. 35
        max_freq = 3*K_BLTZ*temp/H_PLNK
        lsp[freqs > max_freq] = 0.0
        if scalar:
            lsp = np.squeeze(lsp)

        return lsp


def left(yy):
    """
    Mahadevan 1996 Eq.B12
    """
    return yy + 1.852 * np.log(yy)


def left_prime(yy):
    return 1.0 + 1.852/yy


def right(te, msol, fedd, alpha=0.3, c1=0.5, c3=0.3, beta=0.5):
    """
    Mahadevan 1996 Eq.B12
    """

    def lo(the):
        f2 = 0.26*(2.5*np.log(the) - 1/the) + 0.05871
        return f2

    def hi(the):
        f2 = 0.26*(3*np.log(the) + np.log(sp.special.kn(2, 1/the)))
        return f2

    theta_e = K_BLTZ * te / (MELC*SPLC*SPLC)
    APPROX_BELOW = 1.0/30

    f1 = 10.36 + 0.26 * np.log(msol * fedd)
    f3 = 0.26 * np.log(alpha*c1*c3*(1-beta)) + 3.7942

    if np.isscalar(te):
        if (theta_e < APPROX_BELOW):
            f2 = lo(theta_e)
        else:
            f2 = hi(theta_e)
    else:
        f2 = np.zeros_like(theta_e)
        # Use an approximation for small values of `theta_e` (i.e. large arguments to `kn()`)
        idx = (theta_e < APPROX_BELOW)
        f2[idx] = lo(theta_e[idx])
        f2[~idx] = hi(theta_e[~idx])

    return f1 - f2 - f3


def guess_yy(fedd):
    """
    Mahadevan 1996 Eq.B13
    """
    yy = np.power(np.power(10.0, 3.6 + np.log(fedd)/4), 1/3)
    return yy


'''
def freq_from_yy(yy, te, bf):
    """Mahadevan 1996 Eq. 18
    """
    theta_e = K_BLTZ*te / (MELC*SPLC*SPLC)
    xm = np.power(yy, 3)
    vb = QELC * bf / (2*np.pi*MELC*SPLC)
    freq = 3.0*xm*vb*np.square(theta_e)/2.0
    return freq
'''


def xm_from_te(te, mass_sol, fedd):
    # RHS Value we're trying to match
    rr = right(te, mass_sol, fedd)

    if np.isscalar(rr):
        yy = guess_yy(fedd)
        ll = left(yy)

        errs = np.fabs(rr - ll)/rr

        num = 0
        while (errs > 1e-2):
            yy = yy - (ll - rr)/left_prime(yy)
            ll = left(yy)

            errs = np.fabs(rr - ll) / rr

            num += 1
            if num > 100:
                print("cnt = {}, errs = {}".format(num, errs))
                break

    else:
        yy = np.ones_like(rr) * guess_yy(fedd)
        ll = left(yy)

        errs = np.fabs(rr - ll)/rr

        idx = (errs > 1e-2)
        num = 0
        while np.sum(idx) > 0:
            yy[idx] = yy[idx] - (ll[idx] - rr[idx])/left_prime(yy[idx])
            ll[idx] = left(yy[idx])

            errs[idx] = np.fabs(rr[idx] - ll[idx]) / rr[idx]
            idx = (errs > 1e-2)

            num += 1
            if num > 100:
                print("cnt = {}, excess errs = {}".format(num, np.sum(idx)))
                break

    xm = np.power(yy, 3)
    return xm
