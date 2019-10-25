"""
"""

import logging

import numpy as np

import zcode.astro as zastro
import zcode.math as zmath
from zcode.constants import SPLC


class MBH:

    def __init__(self, mass, dist):
        self.mass = mass
        self.dist = dist
        self.rs = zastro.schwarzschild_radius(mass)
        self.rg = self.rs / 2.0
        return


class Disk:

    _PHI_NUM = 40
    _RAD_NUM = 60

    def __init__(self, mbh, ecc=None, inc=None, phi0=None,
                 xsi=[450, 9800], em_gamma=-3.0, sigma_vel=900e5):

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
        self.em_gamma = em_gamma
        self.sigma_vel = sigma_vel

        self.xsi = np.linspace(*xsi, self._RAD_NUM)
        self.rads = self.xsi * self.mbh.rg
        # self.phi = np.linspace(0.0, 2*np.pi, self._PHI_NUM)
        # Shift away from 0.0 to avoid coordinate singularities
        self.phi = np.linspace(0.0, 2*np.pi, self._PHI_NUM, endpoint=False)
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
        '''
        NOTE: I think the above is correct, but the expression for u^alpha (Eq.11)
              is missing a (1+2/x)^{-3/2} in the radial component.
        # Eracleous+1995 - Eq. 16  NOTE: *rederived*
        gamma = np.square(zz*ecc*sinp) + np.square(1 - ecosp)
        gamma /= (zz * xx * (1 - ecosp))
        print("gamma temp = ", zmath.stats_str(gamma))
        print("gamma temp = ", zmath.array_str(gamma.flatten()))
        '''

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

        '''
        # Eracleous+1995 - between eqs 17 and 18
        sigma = self.sigma_vel * freq_rest / SPLC

        # Eracleous+1995 - Eq. 18
        fe = freqs[:, np.newaxis, np.newaxis] / dop[np.newaxis, :, :]
        exp = ((fe - freq_rest)/sigma)**2 / 2

        intens = np.power(xx, self.em_gamma+1)[np.newaxis, :, :] * np.exp(-exp)
        '''
        intens = self.intens(freqs, freq_rest, xx, dop, self.em_gamma)

        flux = freq_rest * np.cos(self.inc) * intens * (dop**3 * psi)[np.newaxis, :, :]

        print("gamma-1 = ", zmath.stats_str(gamma-1))
        print("dop     = ", zmath.stats_str(dop))
        print("dop     = ", zmath.array_str(dop.flatten(), sides=5))
        print("flux  = ", zmath.stats_str(flux))

        return flux

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

        zz = 1.0 - 2.0 / xx

        # Eracleous+1995 - Eq. 9 [four-momentum of a photon]  "p^alpha"
        # NOTE: typo in third component, should be `b r^-2` (negative exponent)
        b_r3 = bor / rads[:, np.newaxis]
        photon = [
            1 / zz,
            - np.sqrt(1 - bor**2 * zz),
            - b_r3,
            0.0
        ]

        # Eqs. 1-2  NOTE: `pp` is "phi-prime" in the text
        #                 this is the expression for "sin(phi)" as a function of "phi-prime"
        sin_phi = np.sin(pp) / np.sqrt(1 - np.square(sini * np.cos(pp)))
        # Eq. 10
        xtil = xx * (1 - ecc * np.cos(pp - self.phi0))
        # Eq 12
        beta_rr = ecc * np.sin(pp - self.phi0) / np.sqrt(xtil)
        # Eq. 13
        beta_pp = (1 - ecc * np.cos(pp - self.phi0)) / np.sqrt(xtil)
        # cc = np.sqrt(zz * xx * (1 - sini_cosp**2))
        # sin_theta = np.sqrt(1 - np.square(np.sin(inc)*np.cos(pp)))
        cosp = np.cos(pp - self.phi0)
        # Eracleous+1995 - Eq. 11 [four-velocity of a disk particle]  "u_alpha"
        # NOTE: error in second component, should have a (1-2/x)^{-3/2} !!
        # NOTE: error in the third component, should have a - sign
        particle = [
            np.sqrt(zz),
            beta_rr / np.power(zz, 3/2),
            - rads[:, np.newaxis] * beta_pp * sini * sin_phi / np.sqrt(zz),
            rads[:, np.newaxis] * np.sqrt((1 - sini**2) * (1 - ecc*cosp) / (xx*zz))
        ]

        gamma = self._gamma_4vel(rads, xx, pp, particle, inc)
        dop = gamma * np.sum([aa*bb for aa, bb in zip(photon, particle)], axis=0)
        dop = 1.0 / dop

        intens = self.intens(freqs, freq_rest, xx, dop, self.em_gamma)

        flux = freq_rest * np.cos(self.inc) * intens * (dop**3 * psi)[np.newaxis, :, :]

        print("gamma-1 = ", zmath.stats_str(gamma-1))
        print("dop     = ", zmath.stats_str(dop))
        print("dop     = ", zmath.array_str(dop.flatten(), sides=5))
        print("flux  = ", zmath.stats_str(flux))

        return flux

    def line_flux_time(self, wave_rest, waves, times, continuum):
        flux = self.line_flux(wave_rest, waves)
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
        # 'roll' (i.e. shift) the continuum by the delay amounts
        cont = zmath.roll(cont, -inds, axis=-1)

        # `cont` shape: (rads, phi, time)    `flux` shape: (freqs, rads, phi)
        cont = np.moveaxis(cont, -1, 0)
        flux = cont[:, np.newaxis, :, :] * flux[np.newaxis, :, :, :]

        return flux
