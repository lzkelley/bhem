"""Advection Dominated Accretion Flow (ADAF) - Accretion disk model.

Based on:
    - "NY95a" : [Narayan & Yi 1995 - 1995ApJ...444..231N]
    - "NY95b" : [Narayan & Yi 1995 - 1995ApJ...452..710N]

"""

import numpy as np

from . import basics, utils
from . constants import NWTG, YR, MSOL, SIGMA_SB, K_BLTZ, H_PLNK, SPLC


class Disk:

    def __init__(self, mass, nrad, mdot=None, fedd=None, rmin=3.0, rmax=1000.0):
        self.mass = mass
        self.mdot, self.fedd = utils.mdot_fedd(mass, mdot, fedd)
        self.ms = mass/MSOL
        self.nrad = nrad
        self.rad_schw = basics.radius_schwarzschild(mass)
        self.rmin = rmin
        self.rmax = rmax

        self._init_primitives()
        self._calc_primitives()

    def __str__(self):
        rv = "Mass: {:.2e} [Msol]\nMdot: {:.1e} [Msol/yr],  Fedd: {:.1e}".format(
            self.ms, self.mdot*YR/MSOL, self.fedd)
        return rv

    def _init_primitives(self):
        nrad = self.nrad

        # Radii
        self.rads = np.logspace(*np.log10([self.rmin, self.rmax]), nrad) * self.rad_schw
        # Density
        self.dens = np.zeros(nrad)
        # Free-fall (dynamical) velocity
        self.vel_ff = np.zeros(nrad)

    def _calc_primitives(self):
        raise RuntimeError("This function must be overwritten")

    @property
    def vel_kep(self):
        """Keplerian velocity profile.

        NY95a Eq. 2.3
        """
        return self.vel_ff/self.rads

    @property
    def rs(self):
        """Radii in schwarzschild units.
        """
        return self.rads/self.rad_schw


class Thin(Disk):

    def __init__(self, mass, nrad, mdot=None, fedd=None, alpha_visc=0.1):
        """
        """

        # Alpha-disc (Shakura-Sunyaev) viscocity parameter
        self.alpha_visc = alpha_visc

        super().__init__(mass, nrad, mdot=mdot, fedd=fedd)

    def _calc_primitives(self):
        """
        """
        mass = self.mass

        # NY95b Eq. 2.2
        self.vel_ff[:] = np.sqrt(NWTG * mass / self.rads)

        self.temp = self._calc_temp_profile()

    def _calc_temp_profile(self):
        """Temperature profile of a thin, optically thick, accretion disk.
        """
        mass = self.mass
        mdot = self.mdot
        rads = self.rads
        rmin = self.rmin

        if np.isscalar(rads):
            temp = 3.0*NWTG*mass*mdot * (1.0 - np.sqrt(rmin/rads))
            temp /= (8*np.pi*np.power(rads, 3.0)*SIGMA_SB)
            temp = np.power(temp, 0.25)
        else:
            temp = np.zeros_like(rads)
            idx = (rads > rmin)
            temp[idx] = 3.0*NWTG*mass*mdot * (1.0 - np.sqrt(rmin/rads[idx]))
            temp[idx] = temp[idx] / (8*np.pi*np.power(rads[idx], 3.0)*SIGMA_SB)
            temp = np.power(temp, 0.25)

        return temp

    def _blackbody_spectral_radiance(self, rads, freqs):
        """The blackbody spectrum of a Shakura-Sunyaev disk as a function of radius and frequency.

        Returns
        -------
        bb : array_like
            Blackbody spectrum as Intensity, [erg/s/Hz/cm^2/steradian]

        """
        denom = np.zeros_like(rads*freqs)

        # Get the temperature profile
        kt = np.broadcast_to(K_BLTZ * self.temp, denom.shape)
        hv = np.broadcast_to(H_PLNK * freqs, denom.shape)

        idx = (kt > 0.0)
        denom[idx] = (np.square(SPLC) * (np.exp(hv[idx]/kt[idx]) - 1.0))
        numer = np.broadcast_to(2.0 * H_PLNK * np.power(freqs, 3.0), denom.shape)
        idx = (denom > 0.0)

        bb = np.zeros_like(denom)
        bb[idx] = numer[idx] / denom[idx]
        return bb

    def blackbody_spectral_luminosity(self, freqs):
        """
        """
        rads = self.rads
        bb_spec_rad = self._blackbody_spectral_radiance(rads[np.newaxis, :], freqs[:, np.newaxis])

        # Integrate over annuli
        annul = np.pi * (np.square(rads[1:]) - np.square(rads[:-1]))
        ave_spec = 0.5 * (bb_spec_rad[:, 1:] + bb_spec_rad[:, :-1])
        bb_lum = 4.0*np.pi * np.sum(ave_spec * annul[np.newaxis, :], axis=-1)
        return bb_lum


class ADAF(Disk):

    def __init__(self, mass, nrad, mdot=None, fedd=None,
                 alpha_visc=0.1, frac_adv=0.5, beta_gp=0.5, frac_hmass=0.75):
        """
        """

        # Alpha-disc (Shakura-Sunyaev) viscocity parameter
        self.alpha_visc = alpha_visc
        # Fraction of viscously dissipated energy which is advected
        self.frac_adv = frac_adv
        # Gas to total pressure fraction (p_g = beta * p)
        self.beta_gp = beta_gp
        # Ratio of specific heats (NY95b Eq 2.7)
        gamma_sh = (32 - 24*beta_gp - 3*np.square(beta_gp)) / (24 - 21*beta_gp)
        self.gamma_sh = gamma_sh

        # Hydrogen mass fraction X
        self.frac_hmass = frac_hmass

        super().__init__(mass, nrad, mdot=mdot, fedd=fedd)

    @property
    def vel_rad(self):
        """Radial velocity.
        """
        # NY95b Eq. 2.1
        return - self._c1 * self.alpha_visc * self.vel_ff

    @property
    def vel_ang(self):
        """Angular velocity.
        """
        # NY95b Eq. 2.1
        return self._c2 * self.vel_ff / self.rads

    @property
    def pres(self):
        # NY95b 2.3
        return self.dens * np.square(self.vel_snd)

    @property
    def pres_mag(self):
        """Magnetic pressure.
        """
        # NY95b Eq. 2.6
        pb = (1 - self.beta_gp) * self.pres
        return pb

    @property
    def pres_gas(self):
        """Gas pressure.
        """
        # NY95b Eq. 2.6
        pg = self.beta_gp * self.pres
        return pg

    @property
    def mag_field_sq(self):
        """Magnetic Field Squared (assuming equipartition).
        """
        # NY95b Eq 2.11
        return 8*np.pi*self.pres_mag

    @property
    def vel_snd(self):
        """Sound speed.
        """
        # NY95b Eq. 2.1
        return np.sqrt(self._c3) * self.vel_ff

    @property
    def visc_diss(self):
        """Viscous dissipation (energy per unit volume).
        """
        # NY95b Eq 2.5
        qplus = 3*self._eps_prime * self.dens * np.fabs(self.vel_rad)
        qplus *= np.square(self.vel_snd) / (2 * self.rads)
        return qplus

    def _calc_primitives(self):
        """

        All equations refer to NY95a
        """
        mass = self.mass
        gamma = self.gamma_sh
        ff = self.frac_adv
        alpha = self.alpha_visc

        # Eq. 2.2
        eps = (5/3 - gamma) / (gamma - 1.0)
        # Eq. 2.20
        eps_prime = eps/ff

        # Eq. 3.4
        gae = np.sqrt(1.0 + 18.0 * np.square(alpha/(5.0 + 2*eps_prime))) - 1.0

        # NY95b Eq. 2.1
        c1 = gae * (5 + 2*eps_prime) / (3 * np.square(alpha))
        c2 = np.sqrt(2 * eps_prime * c1 / 3)
        c3 = 2 * c1 / 3

        x_hmass = self.frac_hmass
        molw_ion = 4 / (1 + 3*x_hmass)
        molw_elec = 2 / (1 + x_hmass)

        # Store
        self._gae = gae
        self._eps = eps
        self._eps_prime = eps_prime
        self._c1 = c1
        self._c2 = c2
        self._c3 = c3
        self.molw_ion = molw_ion
        self.molw_elec = molw_elec

        # NY95b Eq. 2.2
        self.vel_ff[:] = np.sqrt(NWTG * mass / self.rads)
        # NY95b Eq. 2.1
        self.vel_rad[:] = - c1 * alpha * self.vel_ff[:]

        # NY95b Eq. 2.3
        self.dens[:] = (self.mdot /
                        (4*np.pi*np.sqrt(2.5*c3) * np.square(self.rads) * np.fabs(self.vel_rad)))
