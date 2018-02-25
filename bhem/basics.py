"""Basic relations and calculations.
"""

import numpy as np
# import scipy as sp
# import scipy.integrate

from . import constants
from . constants import NWTG, SIGMA_SB, SPLC, SIGMA_T, MPRT, H_PLNK, K_BLTZ

_SCHW_CONST = 2.0*NWTG/np.square(SPLC)          # for Schwarzschild radius
_EDD_CONST = 4.0*np.pi*SPLC*NWTG*MPRT/SIGMA_T   # for Eddington accretion/luminosity

RAD_INNER = 6.0      # Inner disk edge in units of Schwarzschild radius
RAD_OUTER = 1.0e4   # Outer disk edge in units of Schwarzschild radius


def _get_rad_inner(mass, ri=None):
    if ri is None:
        ri = radius_schwarzschild(mass) * RAD_INNER
    return ri


def _get_rad_outer(mass, ro=None):
    if ro is None:
        ro = radius_schwarzschild(mass) * RAD_OUTER
    return ro


def temperature_profile(mass, mdot, rads, ri=None):
    """Temperature profile of a thin, optically thick, accretion disk.
    """
    ri = _get_rad_inner(mass, ri)

    if np.isscalar(rads):
        temp = 3.0*NWTG*mass*mdot * (1.0 - np.sqrt(ri/rads))
        temp /= (8*np.pi*np.power(rads, 3.0)*SIGMA_SB)
        temp = np.power(temp, 0.25)
    else:
        temp = np.zeros_like(rads)
        idx = (rads > ri)
        temp[idx] = 3.0*NWTG*mass*mdot * (1.0 - np.sqrt(ri/rads[idx]))
        temp[idx] = temp[idx] / (8*np.pi*np.power(rads[idx], 3.0)*SIGMA_SB)
        temp = np.power(temp, 0.25)

    return temp


def radius_schwarzschild(mass):
    """
    """
    rsch = mass * _SCHW_CONST   # 2*G/c^2
    return rsch


def eddington_luminosity(mass):
    """
    """
    lum = mass * _EDD_CONST
    return lum


def eddington_accretion(mass, eps=0.1):
    """Eddington Accretion rate, $\dot{M}_{Edd} = L_{Edd}/\epsilon c^2$.^

    Arguments
    ---------
    mass : array_like of scalar
        BH Mass.
    eps : array_like of scalar
        Efficiency parameter.

    Returns
    -------
    mdot : array_like of scalar
        Eddington accretion rate.

    """
    mdot = (mass * _EDD_CONST) / (eps * np.square(SPLC))
    return mdot


def blackbody_spectral_radiance(mass, mdot, rads, freq, ri=None):
    """The blackbody spectrum of a Shakura-Sunyaev disk as a function of radius and frequency.

    Returns
    -------
    bb : array_like
        Blackbody spectrum as Intensity, [erg/s/Hz/cm^2/steradian]

    """

    denom = np.zeros_like(rads*freq)

    # Get the temperature profile
    kt = np.broadcast_to(K_BLTZ * temperature_profile(mass, mdot, rads, ri=ri), denom.shape)
    hv = np.broadcast_to(H_PLNK * freq, denom.shape)

    idx = (kt > 0.0)
    denom[idx] = (np.square(SPLC) * (np.exp(hv[idx]/kt[idx]) - 1.0))
    numer = np.broadcast_to(2.0 * H_PLNK * np.power(freq, 3.0), denom.shape)
    idx = (denom > 0.0)

    bb = np.zeros_like(denom)
    bb[idx] = numer[idx] / denom[idx]
    return bb


def blackbody_spectral_luminosity(mass, mdot, freq, ri=None, ro=None):
    """
    """
    ri = _get_rad_inner(mass, ri)
    ro = _get_rad_outer(mass, ro)

    # Calculate number of resolution elements based on number of decades from inner to outer radii
    NUM_PER_DEC = 40
    ndec = np.int(np.ceil(np.log10(ro/ri)) * NUM_PER_DEC)
    ndec = np.min([ndec, 1000])
    ndec = np.max([ndec, 100])
    # Construct radial points
    rads = np.logspace(*np.log10([ri, ro]), ndec)

    bb_spec_rad = blackbody_spectral_radiance(mass, mdot, rads[np.newaxis, :], freq[:, np.newaxis])

    # Integrate over annuli
    annul = np.pi * (np.square(rads[1:]) - np.square(rads[:-1]))
    ave_spec = 0.5 * (bb_spec_rad[:, 1:] + bb_spec_rad[:, :-1])
    bb_lum = 4.0*np.pi * np.sum(ave_spec * annul[np.newaxis, :], axis=-1)
    return bb_lum
