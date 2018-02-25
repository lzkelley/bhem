"""Advection Dominated Accretion Flow (ADAF) - Accretion disk model.

Based on:
    - "NY95a" : [Narayan & Yi 1995 - 1995ApJ...444..231N]
    - "NY95b" : [Narayan & Yi 1995 - 1995ApJ...452..710N]

"""

import numpy as np

from . import basics
from . constants import SPLC, NWTG

# Extrema in radii, in units of Schwarzschild
RAD_EXTR = [1.0, 1.0e4]


class Disk:

    def __init__(self, mass, mdot, nrad):
        self.mass = mass
        self.mdot = mdot
        self.nrad = nrad
        self.rad_schw = basics.radius_schwarzschild(mass)

        self._init_primitives()

        self._calc_primitives()

    def __str__(self):
        pass

    def _init_primitives(self):
        nrad = self.nrad

        # Radii
        self.rads = np.logspace(*np.log10(RAD_EXTR), nrad) * self.rad_schw
        # Density
        self.dens = np.zeros(nrad)
        # Free-fall (dynamical) velocity
        self.vel_ff = np.zeros(nrad)

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


class ADAF(Disk):

    def __init__(self, mass, mdot, nrad, alpha_visc=0.1, frac_adv=0.5, gamma_sh=4.0/3.0):
        super.__init__(mass, mdot, nrad)
        # Alpha-disc (Shakura-Sunyaev) viscocity parameter
        self.alpha_visc = alpha_visc
        # Fraction of viscously dissipated energy which is advected
        self.frac_adv = frac_adv
        # Ratio of specific heats
        self.gamma_sh = gamma_sh

    @property
    def vel_rad(self):
        """Radial velocity.
        """
        # NY95b Eq. 2.1
        return - self.c1 * self.alpha_visc * self.vel_ff

    @property
    def vel_ang(self):
        """Angular velocity.
        """
        # NY95b Eq. 2.1
        return self.c2 * self.vel_ff / self.rads

    @property
    def pres(self):
        # NY95b 2.3
        return self.dens * np.square(self.vel_snd)

    @property
    def vel_snd(self):
        """Sound speed.
        """
        # NY95b Eq. 2.1
        return np.sqrt(self.c3) * self.vel_ff

    def _calc_primitives(self):
        """

        All equations refer to NY95a
        """
        mass = self.mass
        gamma = self.gamma_sh
        ff = self.frac_adv
        alpha = self.alpha_visc

        # Eq. 2.2
        eps = 5.0/3.0 - gamma / (gamma - 1.0)
        # Eq. 2.20
        eps_prime = eps/ff

        # Eq. 3.4
        gae = np.sqrt(1.0 - 18.0 * np.square(alpha/(5.0 + 2*eps_prime))) - 1.0

        # NY95b Eq. 2.1
        c1 = - gae * (5 + 2*eps_prime) / (3 * np.square(alpha))
        c2 = np.sqrt(2 * eps_prime * c1 / 3)
        c3 = 2 * c1 / 3

        # NY95b Eq. 2.2
        self.vel_ff[:] = np.sqrt(NWTG * mass / self.rads)
        # NY95b Eq. 2.1
        self.vel_rad[:] = - c1 * alpha * self.vel_ff[:]

        # NY95b Eq. 2.3
        self.dens[:] = (self.mdot /
                        (4*np.pi*np.sqrt(2.5*c3) * np.square(self.rads) * np.fabs(self.vel_rad)))

        # Store
        self._gae = gae
        self._eps = eps
        self._eps_prime = eps_prime
        self._c1 = c1
        self._c2 = c2
        self._c3 = c3
