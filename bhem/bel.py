"""
"""

'''
import logging

import numpy as np
import scipy as sp
import scipy.optimize   # noqa

import zcode.astro as zastro
import zcode.math as zmath
from zcode.constants import SPLC, DAY, MSOL, PC

SIGMA_TO_FWHM = 2*np.sqrt(2*np.log(2.0))

import bhem


class MBH:

    def __init__(self, mass, fedd, dist):
        self.mass = mass
        self.fedd = fedd
        self.dist = dist
        self.rs = zastro.schwarzschild_radius(mass)
        self.rg = self.rs / 2.0
        return

    def drw_lightcurve(self, times, tau=None, mean_mag=None, sfinf=None, size=None):
        _imag, _tau, _sfinf = bhem.drw.drw_params(self.mass, self.fedd, samples=False)
        if tau is None:
            tau = _tau
        if mean_mag is None:
            mean_mag = _imag
        if sfinf is None:
            sfinf = _sfinf

        lums = bhem.drw.drw_lightcurve(times, tau, mean_mag, sfinf, size=size)

        return lums


class Line_Time_Spectrum:

    def __init__(self, disk, times, waves, flux, continuum, wave_rest):
        self._disk = disk

        self.times = times
        self.waves = waves
        self.flux = flux
        self.continuum = continuum
        self.wave_rest = wave_rest
        return

    @property
    def flux_twrp(self):
        return self.flux

    @property
    def flux_tw(self):
        try:
            return self._flux_tw
        except AttributeError:
            pass

        _flux_tw = np.trapz(self.flux_twrp, x=self._disk.phi, axis=-1)
        _flux_tw = np.trapz(_flux_tw, x=self._disk.xsi, axis=-1)
        self._flux_tw = _flux_tw
        return _flux_tw

    @property
    def flux_t(self):
        try:
            return self._flux_t
        except AttributeError:
            pass

        _flux_t = np.trapz(self.flux_tw, x=self.waves, axis=-1)
        self._flux_t = _flux_t
        return _flux_t

    @property
    def red(self):
        try:
            return self._red
        except AttributeError:
            self._construct_twin_peaks()

        return self._red

    @property
    def wht(self):
        try:
            return self._wht
        except AttributeError:
            self._construct_twin_peaks()

        return self._wht

    @property
    def blu(self):
        try:
            return self._blu
        except AttributeError:
            self._construct_twin_peaks()

        return self._blu

    def dv_stats(self):
        """Ave and Std of the velocity-offset for red,wht,blu components
        """
        comps = [self.red, self.wht, self.blu]
        # (3, 2)
        dv = [[np.average(xx[:, 1]), np.std(xx[:, 1])] for xx in comps]
        return dv

    def _construct_twin_peaks(self):
        ntimes = self.times.size
        # flux[time, wavelength]
        zz = self.flux_tw
        zz = zz / zz.max()
        # Wavelength in Angstroms
        ww = self.waves * 1e8
        w0 = self.wave_rest * 1e8

        # Amplitude, location, sigma
        red = np.zeros((ntimes, 3))
        blu = np.zeros((ntimes, 3))
        wht = np.zeros((ntimes, 3))
        pars = np.zeros((ntimes, 6))

        for ii, yy in enumerate(zz):
            norm = yy.max()
            half_max, pars[ii, :] = fit_twin_peaks(ww, yy/norm)

            # Integrated flux
            wht[ii, 0] = np.trapz(yy, x=ww)
            # FWHM
            wht[ii, 2] = np.diff(half_max)
            # Centroid location
            # wht[ii, 1] = np.average(ww, weights=yy)
            wht[ii, 1] = np.average(half_max)

            # Red/Blue Components
            blu[ii, :] = pars[ii, :3]
            red[ii, :] = pars[ii, 3:]
            blu[ii, 0] = blu[ii, 0] * norm
            red[ii, 0] = red[ii, 0] * norm

        # convert from sigma to FWHM
        blu[:, -1] = blu[:, -1] * SIGMA_TO_FWHM
        red[:, -1] = red[:, -1] * SIGMA_TO_FWHM

        # Convert from wavelengths [Angstrom] to velocities [km/s]
        blu[:, 1] = lambda_to_dvel(blu[:, 1], w0, vunits=1e5)
        red[:, 1] = lambda_to_dvel(red[:, 1], w0, vunits=1e5)
        wht[:, 1] = lambda_to_dvel(wht[:, 1], w0, vunits=1e5)

        blu[:, 2] = (SPLC / 1e5) * (blu[:, 2] / w0)
        red[:, 2] = (SPLC / 1e5) * (red[:, 2] / w0)
        wht[:, 2] = (SPLC / 1e5) * (wht[:, 2] / w0)

        self._red = red
        self._wht = wht
        self._blu = blu
        self._pars = pars

        return


class Disk:

    def __init__(self, mbh, ecc=None, inc=None, phi0=None, frad=0.2,
                 xsi=[450, 9800], em_gamma=-3.0, sigma_vel=900e5, num_rad=80, num_phi=90):

        if inc is None:
            inc = np.deg2rad(27)
        if phi0 is None:
            phi0 = np.deg2rad(220)
        if ecc is None:
            ecc = 0.5

        self.mbh = mbh
        self.inc = inc
        self.phi0 = phi0
        self.ecc = ecc
        self.frad = frad
        self.em_gamma = em_gamma
        self.sigma_vel = sigma_vel

        self.xsi = np.linspace(*xsi, num_rad)
        self.rads = self.xsi * self.mbh.rg
        # self.phi = np.linspace(0.0, 2*np.pi, self._PHI_NUM)
        # Shift away from 0.0 to avoid coordinate singularities
        self.phi = np.linspace(0.0, 2*np.pi, num_phi, endpoint=False)
        dp = (2*np.pi - self.phi[-1]) / 2.0
        self.phi += dp
        return

    def delay(self):

        dist = self.mbh.dist
        phi = self.phi
        sini = np.sin(self.inc)
        rads = self.rads
        rads, phi = zmath.broadcast(rads, phi)

        cosp = np.cos(phi)
        # ll = dist**2 * (1 + (rads*sini/dist)**2 - 2*(rads*sini*cosp/dist))

        delay = dist**2 - rads**2 * sini - 2*rads*dist*sini*cosp
        delay = rads + np.sqrt(delay) - dist

        delay /= SPLC
        return delay

    def _psi_bor(self, xx, pp, ecc, inc):
        ecc = ecc
        sini = np.sin(inc)
        sini_cosp = sini * np.cos(pp)

        # Eracleous+1995 - Eq. 8
        psi = (1 - sini_cosp) / (1 + sini_cosp)
        psi = 1 + psi / xx

        # Eracleous+1995 - Eq. 6
        bor = np.sqrt(1 - sini_cosp**2) * psi
        return psi, bor, sini, sini_cosp

    def line_flux_e95(self, wave_rest, waves):

        freq_rest = SPLC / wave_rest
        freqs = SPLC / waves

        # Broadcast to (X, P)
        xx, pp = zmath.broadcast(self.xsi, self.phi)

        ecc = self.ecc
        psi, bor, sini, sini_cosp = self._psi_bor(xx, pp, ecc, self.inc)

        sinp = np.sin(pp - self.phi0)
        cosp = np.cos(pp - self.phi0)
        ecosp = ecc * cosp

        zz = (1 - 2/xx)
        # Eracleous+1995 - Eq. 16 as given  NOTE: possible errors (actually, I think it's fine)
        a1 = np.square(ecc * sinp)
        a2 = zz * np.square(1 - ecosp)
        a3 = xx * np.square(zz) * (1 - ecosp)
        gamma = (a1 + a2) / a3
        """
        NOTE: I think the above is correct, but the expression for u^alpha (Eq.11)
              is missing a (1+2/x)^{-3/2} in the radial component.
        # Eracleous+1995 - Eq. 16  NOTE: *rederived*
        gamma = np.square(zz*ecc*sinp) + np.square(1 - ecosp)
        gamma /= (zz * xx * (1 - ecosp))
        print("gamma temp = ", zmath.stats_str(gamma))
        print("gamma temp = ", zmath.array_str(gamma.flatten()))
        """

        gamma = 1 / np.sqrt(1 - gamma)

        b1 = np.sqrt(1 - bor**2 * zz)*ecc*sinp
        b2 = np.sqrt(xx) * np.power(zz, 3/2) * np.sqrt(1 - ecosp)   # as given in Eracleous
        # b2 = np.sqrt(xx) * np.sqrt(1 - ecosp)    # fixed error?? (maybe?)
        b3 = bor * np.sqrt(1 - ecosp) * sini*np.sin(pp)
        b4 = np.sqrt(xx*zz*(1 - sini**2 * np.cos(pp)**2))

        # Eracleous+1995 - Eq. 15 - as written  *NOTE: may have error (`+` should be `-`)
        dop = (1/np.sqrt(zz) - b1/b2 + b3/b4)
        # Eracleous+1995 - Eq. 15 -  *NOTE: rederived*
        # dop = (1/np.sqrt(zz) - b1/b2 - b3/b4)
        dop = 1 / (gamma * dop)

        """
        # Eracleous+1995 - between eqs 17 and 18
        sigma = self.sigma_vel * freq_rest / SPLC

        # Eracleous+1995 - Eq. 18
        fe = freqs[:, np.newaxis, np.newaxis] / dop[np.newaxis, :, :]
        exp = ((fe - freq_rest)/sigma)**2 / 2

        intens = np.power(xx, self.em_gamma+1)[np.newaxis, :, :] * np.exp(-exp)
        """
        intens = self.intens(freqs, freq_rest, xx, dop, self.em_gamma)

        flux = freq_rest * np.cos(self.inc) * intens * (dop**3 * psi)[np.newaxis, :, :]

        # print("gamma-1 = ", zmath.stats_str(gamma-1))
        # print("dop     = ", zmath.stats_str(dop))
        # print("dop     = ", zmath.array_str(dop.flatten(), sides=5))
        # print("flux  = ", zmath.stats_str(flux))

        return flux, gamma, dop, intens

    def line_flux_e95_fix(self, wave_rest, waves):

        freq_rest = SPLC / wave_rest
        freqs = SPLC / waves

        # Broadcast to (X, P)
        xx, pp = zmath.broadcast(self.xsi, self.phi)

        ecc = self.ecc
        psi, bor, sini, sini_cosp = self._psi_bor(xx, pp, ecc, self.inc)

        sinp = np.sin(pp - self.phi0)
        cosp = np.cos(pp - self.phi0)
        ecosp = ecc * cosp

        chi = (1 - 2/xx)
        kappa = 1 - ecosp
        # Eracleous+1995 - Eq. 16, but "fixed" according to my derivation
        gamma = (np.square(ecc * sinp) + kappa**2) / (xx*kappa)
        gamma = 1 / np.sqrt(1 - gamma)

        # Eracleous+1995 - Eq. 15, but "fixed" according to rederivation
        d1 = ecc*sinp*np.sqrt((1 - chi*bor**2) / (xx*kappa))
        d2 = bor*sini*np.sin(pp)*np.sqrt(kappa / (xx*(1 - sini_cosp**2)))
        dop = (1 - d1 + d2)
        # Eracleous+1995 - Eq. 15 -  *NOTE: rederived*
        # dop = (1/np.sqrt(zz) - b1/b2 - b3/b4)
        dop = np.sqrt(chi) / (gamma * dop)

        intens = self.intens(freqs, freq_rest, xx, dop, self.em_gamma)

        flux = freq_rest * np.cos(self.inc) * intens * (dop**3 * psi)[np.newaxis, :, :]

        # print("gamma-1 = ", zmath.stats_str(gamma-1))
        # print("dop     = ", zmath.stats_str(dop))
        # print("dop     = ", zmath.array_str(dop.flatten(), sides=5))
        # print("flux  = ", zmath.stats_str(flux))

        return flux, gamma, dop, intens

    def _gamma_4vel(self, rads, xx, pp, vel, inc):
        """Calculate `gamma` given the last three components of a four-velocity

        Arguments
        ---------
        rr : (R,) radii in physical units
        xx : (R, P,) dimensionless radii (in units of gravitational-radius)
        pp : (R, P,) phi angles

        """
        zz = 1.0 - 2.0/xx
        r2 = rads[:, np.newaxis] ** 2

        # ecc = self.ecc
        sini = np.sin(inc)
        # sinp = np.sin(pp - self.phi0)
        # cosp = np.cos(pp - self.phi0)
        # sin_phi = np.sin(pp) / np.sqrt(1 - np.square(sini*np.cos(pp)))
        # omecp = (1 - ecc*cosp)

        sin_theta2 = (1 - np.square(sini*np.cos(pp)))

        # Construct the contravariant metric (g^{alpha beta}), [t, r, theta, phi]
        gg = [-1/zz, zz, 1/r2, 1/(r2*sin_theta2)]

        # Eracleous+1995 - Eq. 14
        gamma = [v*g*v for (g, v) in zip(gg[1:], vel[1:])]

        gamma = np.sum(gamma, axis=0)
        # print("gamma temp = ", zmath.stats_str(gamma))
        # print("gamma temp = ", zmath.array_str(gamma.flatten()))
        gamma = 1/np.sqrt(1.0 - gamma)

        return gamma

    def intens(self, freqs, freq_rest, xx, dop, em_gamma):
        # Eracleous+1995 - between eqs 17 and 18
        sigma = self.sigma_vel * freq_rest / SPLC

        # Eracleous+1995 - Eq. 18
        fe = freqs[:, np.newaxis, np.newaxis] / dop[np.newaxis, :, :]
        exp = ((fe - freq_rest)/sigma)**2 / 2
        intens = np.power(xx, em_gamma+1)[np.newaxis, :, :] * np.exp(-exp)

        return intens

    def line_flux(self, wave_rest, waves):
        freq_rest = SPLC / wave_rest
        freqs = SPLC / waves

        # (R, P)
        xx, pp = zmath.broadcast(self.xsi, self.phi)

        ecc = self.ecc
        inc = self.inc
        rads = self.rads
        psi, bor, sini, sini_cosp = self._psi_bor(xx, pp, ecc, inc)

        chi = 1.0 - 2.0 / xx

        # Eracleous+1995 - Eq. 9 [four-momentum of a photon]  "p^alpha"
        # NOTE: typo in third component, should be `b r^-2` (negative exponent)
        photon = [
            1 / chi,
            - np.sqrt(1 - bor**2 * chi),
            - bor / rads[:, np.newaxis],
            0.0
        ]

        # 2th component (i.e. third) can be nan (from rounding error?)
        photon = [np.nan_to_num(pp) for pp in photon]

        # Eqs. 1-2  NOTE: `pp` is "phi-prime" in the text
        #                 this is the expression for "sin(phi)" as a function of "phi-prime"
        sin_phi = np.sin(pp) / np.sqrt(1 - np.square(sini * np.cos(pp)))
        kappa = (1 - ecc * np.cos(pp - self.phi0))
        # Eq. 10
        xtil = xx * kappa
        # Eq 12
        beta_rr = ecc * np.sin(pp - self.phi0) / np.sqrt(xtil)
        # Add in an additional radial component, `f_r (r_s/r)^{1/2}`
        beta_rr += self.frad * np.sqrt(2.0 / xx)

        # Eq. 13
        # beta_pp = kappa / np.sqrt(xtil)
        # cosp = np.cos(pp - self.phi0)
        # Eracleous+1995 - Eq. 11 [four-velocity of a disk particle]  "u_alpha / gamma"
        # NOTE: error in second component, should have a (1-2/x)^{-3/2} !! MAYBE
        # NOTE: error in the third component, should have a - sign
        # particle = [
        #     np.sqrt(chi),
        #     beta_rr / np.power(chi, 3/2),
        #     - rads[:, np.newaxis] * beta_pp * sini * sin_phi / np.sqrt(chi),
        #     rads[:, np.newaxis] * np.sqrt((1 - sini**2) * (1 - ecc*cosp) / (xx*chi))
        # ]
        particle = [
            np.sqrt(chi),
            beta_rr / np.sqrt(chi),
            - rads[:, np.newaxis] * sini * sin_phi * np.sqrt(kappa / xx),
            rads[:, np.newaxis] * sini * np.sqrt((1 - np.square(sini*sin_phi))*kappa / xx)
            # rads[:, np.newaxis] * np.sqrt((1 - sini**2) * (1 - ecc*cosp) / (xx*chi))
        ]

        gamma = self._gamma_4vel(rads, xx, pp, particle, inc)
        dop = gamma * np.sum([aa*bb for aa, bb in zip(photon, particle)], axis=0)
        # print("dop     = ", zmath.stats_str(dop))
        # print("dop     = ", zmath.array_str(dop.flatten(), sides=5))
        dop = 1.0 / dop
        # print("\tdop     = ", zmath.stats_str(dop))
        # print("\tdop     = ", zmath.array_str(dop.flatten(), sides=5))

        intens = self.intens(freqs, freq_rest, xx, dop, self.em_gamma)

        flux = freq_rest * np.cos(self.inc) * intens * (dop**3 * psi)[np.newaxis, :, :]

        # print("gamma-1 = ", zmath.stats_str(gamma-1))
        # print("dop     = ", zmath.stats_str(dop))
        # print("dop     = ", zmath.array_str(dop.flatten(), sides=5))
        # print("flux  = ", zmath.stats_str(flux))

        return flux, gamma, dop, intens

    def line_flux_time(self, wave_rest, waves, times, continuum):
        flux, *_ = self.line_flux(wave_rest, waves)
        delay = self.delay()

        if delay.max() >= times[-1]:
            err = "Warning delays [{}, {}] exceed times [{}, {}]!".format(
                *zmath.minmax(delay), *zmath.minmax(times))
            logging.warning(err)

        # Shift the continuum emission by each delay
        inds = zmath.argnearest(times, delay)

        # `continuum` is shape (times,) broadcast to (rads, phi, times)
        shape = (self.xsi.size, self.phi.size, continuum.size)
        cont = np.broadcast_to(continuum, shape)
        # Repeat the continuum lightcurve a 2nd time, but shift it to match up with the end of 1st
        repeat = (cont[..., -1] - cont[..., 0])[..., np.newaxis] + cont[..., 1:]

        # 'roll' (i.e. shift) the continuum by the delay amounts
        cont = zmath.roll(cont, -inds, cat=repeat, axis=-1)

        # `cont` shape: (rads, phi, time)    `flux` shape: (freqs, rads, phi)
        cont = np.moveaxis(cont, -1, 0)
        flux = cont[:, np.newaxis, :, :] * flux[np.newaxis, :, :, :]
        # ==> shape: (time, freqs, rads, phi)

        lts = Line_Time_Spectrum(self, times, waves, flux, continuum, wave_rest)

        return lts

    """
    def line_flux_time_nd(self, wave_rest, waves, times, continuum):
        flux, *_ = self.line_flux(wave_rest, waves)
        delay = self.delay()
        nrads, nphi = delay.shape

        if delay.max() >= times[-1]:
            err = "Warning delays [{}, {}] exceed times [{}, {}]!".format(
                *zmath.minmax(delay), *zmath.minmax(times))
            logging.warning(err)

        # Shift the continuum emission by each delay
        inds = zmath.argnearest(times, delay)
        ntimes, nreals = continuum.shape
        inds = np.broadcast_to(inds, (nreals, nrads, nphi))
        inds = np.moveaxis(inds, 0, -1)

        # `continuum` is shape (times,) broadcast to (rads, phi, times)
        shape = (self.xsi.size, self.phi.size,) + continuum.shape
        print("shape = ", shape)
        cont = np.broadcast_to(continuum, shape)
        # Repeat the continuum lightcurve a 2nd time, but shift it to match up with the end of 1st
        repeat = (cont[..., -1, :] - cont[..., 0, :])[..., np.newaxis, :] + cont[..., 1:, :]

        # 'roll' (i.e. shift) the continuum by the delay amounts
        cont = zmath.roll(cont, -inds, cat=repeat, axis=-2)

        # `cont` shape: (rads, phi, time)    `flux` shape: (freqs, rads, phi)
        cont = np.moveaxis(cont, -2, 0)
        flux = cont[:, np.newaxis, :, :, :] * flux[np.newaxis, :, :, :, np.newaxis]
        print("flux shape = ", flux.shape)
        # ==> shape: (time, freqs, rads, phi)

        lts = Line_Time_Spectrum(self, times, waves, flux, continuum, wave_rest)

        return lts
    """


def half_max_locs(waves, flux):
    locs = np.zeros(2)
    num = flux.size-1
    side = 2

    for jj, cut in enumerate([slice(None), slice(None, None, -1)]):
        ff = flux[cut]/flux.max()
        ww = waves[cut]
        kk = np.argmax(ff > 0.5)
        if kk <= 0:
            locs[jj] = ww[0]
            continue
        elif kk >= num:
            locs[jj] = ww[-1]
            continue

        lo = np.max([0, kk-side])
        hi = np.min([num, kk+(side+1)])
        if lo == 0:
            hi += (lo - (kk-side))
        if hi == num:
            lo += (hi - (kk+side+1))

        if (hi - lo) != (2*side + 1):
            err = "Failed to construct cut!  kk = {}, lo = {}, hi = {}!".format(
                kk, lo, hi)
            raise ValueError(err)

        cut = slice(lo, hi)

        locs[jj] = sp.interpolate.interp1d(ff[cut], ww[cut], kind='quadratic')(0.5)

    return locs


def twin_peaks_each(waves, a1, l1, w1, a2, l2, w2):
    pks = np.zeros((waves.size, 2))
    for ii, (aa, ll, ww) in enumerate(zip([a1, a2], [l1, l2], [w1, w2])):
        pks[:, ii] = aa * np.exp(-np.square((waves - ll)/ww)/2)

    return pks


def twin_peaks(*args):
    pks = twin_peaks_each(*args)
    return np.sum(pks, axis=-1)


def guess_twin_peaks(xx, yy):
    cent = np.average(xx, weights=yy)
    half_maxes = half_max_locs(xx, yy)
    locs = [np.average([hm, cent]) for hm in half_maxes]
    amps = [np.sum((yy[(comp(xx, cent) & comp(hm, xx))])) for comp, hm in zip([np.less, np.greater], half_maxes)]
    amps = [1.0, amps[1]/amps[0]]
    wids = [np.fabs(ll - cent)*0.75 for ll in locs]
    pars = [amps[0], locs[0], wids[0], amps[1], locs[1], wids[1]]
    return half_maxes, pars


def fit_twin_peaks(xx, yy):
    sigma = fake_errors(xx, yy)
    half_max, guess_pars = guess_twin_peaks(xx, yy)
    try:
        popt, pcov = sp.optimize.curve_fit(twin_peaks, xx, yy, p0=guess_pars, sigma=sigma)
    except (ValueError, RuntimeError):
        print("WARNING: `sp.optimize.curve_fit` failed!")
        popt = np.zeros(6)

    return half_max, popt


def fake_errors(xx, yy):
    sigma_min = 0.025
    sigma = sigma_min*np.power(yy, -2.0)
    sigma = np.clip(sigma, sigma_min, 0.3)
    return sigma


def lambda_to_dvel(waves, reference, vunits=1e5):
    if reference is not None:
        dl_l = (waves - reference) / reference
    else:
        dl_l = waves

    dv = (SPLC / vunits) * dl_l
    return dv


def iband_from_mass_fedd(mass, fedd, eps=1.0):
    lbol = zastro.eddington_luminosity(mass, eps) * fedd
    l5100 = bhem.obs.lum5100_from_lbol_runnoe2012(lbol)
    L_lambda = l5100 * (5100e-8)
    imag = zastro.obs.lum_to_abs_mag('i', L_lambda, type='l')
    return imag


def rad_blr(mass, fedd):
    vv = (mass / (1e8*MSOL)) * (fedd / 0.1)
    rb = 0.16*PC * np.power(vv, 0.59)
    return rb
'''
