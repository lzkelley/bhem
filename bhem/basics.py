"""Basic relations and calculations.
"""

import numpy as np
# import scipy as sp
# import scipy.integrate

# from . import constants
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
