"""Radiation Calculations
"""

import numpy as np
import scipy as sp
import scipy.special

from . constants import SPLC, MELC, MPRT, K_BLTZ, RELC

CSQ = np.square(SPLC)
MELC_C2 = MELC * CSQ
MPRT_C2 = MPRT * CSQ


def heating_coulomb_ie(ne, ni, te, ti):
    """Heating of electrons from Coulomb collisions with Ions.

    Based on NY95b Eq 3.1

    The Bessel functions (`K_n(x)`) used in this calculated approach zero *very quickly*
    for `x >> 1`.  This causes numerical issues when using the expression as given (Eq. 3.1).
    Instead, in problematic regimes, use approximation for large x, i.e. that

        lim x->inf  K_n(x) = a * exp(-x) / sqrt(x),   where  a = sqrt(pi/2)

    This approach seems to be accurate to at worst about 10%.

    """

    if np.isscalar(te) or np.isscalar(ti):
        raise ValueError("Only arrays work as temperature input ATM")

    NORM = 5.61e-32  # erg/cm^3/s

    # where to use approximations for Bessel functions
    #    Note that the arguments to bessel functions are ~ 1/theta (the dimensionless temperatures)
    CUTOFF = 1.0/30.0

    kte = K_BLTZ * te
    kti = K_BLTZ * ti

    # Dimensionless temperatures (NY95b Eq. 3.2)
    theta_e = kte / MELC_C2
    theta_i = kti / MPRT_C2
    te_ti = theta_e + theta_i
    red_temp = te_ti / (theta_e * theta_i)
    term = ((np.square(te_ti) + 1) / te_ti) + 1

    k2e = sp.special.kn(2, 1/theta_e)
    k2i = sp.special.kn(2, 1/theta_i)

    k1r = sp.special.kn(1, red_temp)
    k0r = sp.special.kn(0, red_temp)

    # We have a general expression below, `qie = amp * f1 * (term * f2 + f3)`
    #    where `amp` and `term` are fixed, but `f1`, `f2`, `f3` may be calculated using
    #    approximations when in the appropriate regime
    f1 = np.ones_like(te)
    f2 = np.ones_like(te)
    f3 = np.ones_like(te)

    # If both dimensionless-temperatures are <~ 1, use full expression
    idx = (theta_e >= CUTOFF) & (theta_i >= CUTOFF)
    f1[idx] = 1.0 / (k2e[idx] * k2i[idx])
    f2[idx] = k1r[idx]
    f3[idx] = k0r[idx]

    # If both dimensionless-temperatures are >> 1, use full approximation
    #    f2 and f3 are unity in this case
    idx = (theta_e < CUTOFF) & (theta_i < CUTOFF)
    f1[idx] = np.sqrt(2 / (np.pi * te_ti[idx]))

    # If electron dimensionless-temp is >> 1, approximate only that term
    #    f2 and f3 are unity in this case
    idx = (theta_e < CUTOFF) & (theta_i >= CUTOFF)
    f1[idx] = np.exp(-1/theta_i[idx]) / k2i[idx]

    # If ion dimensionless-temp is >> 1, approximate only that term
    #    f2 and f3 are unity in this case
    idx = (theta_e >= CUTOFF) & (theta_i < CUTOFF)
    f1[idx] = np.exp(-1/theta_e[idx]) / k2e[idx]

    amp = NORM * 2 * ne * ni * (ti - te)
    qie = amp * f1 * (term * f2 + f3)
    return qie


def cooling_brems_ei(ne, te):
    """Cooling due to Bremsstrahlung from electron-ion scattering.

    NY95b Eq. 3.5
    """
    NORM = 1.48e-22   # erg/cm^3/sec
    theta_e = K_BLTZ * te / MELC_C2

    def func_lo(th_e):
        return 4 * np.sqrt(2*th_e/np.power(np.pi, 3)) * (1 + 1.781 * np.power(th_e, 1.34))

    def func_hi(th_e):
        return 9 * th_e * (np.log(1.123*th_e + 0.48) + 1.5) / 2*np.pi

    # eq. 3.6
    if np.isscalar(theta_e):
        if theta_e <= 1.0:
            fei = func_lo(theta_e)
        else:
            fei = func_hi(theta_e)
    else:
        fei = np.zeros_like(theta_e)
        idx = (theta_e <= 1.0)
        fei[idx] = func_lo(theta_e[idx])
        fei[~idx] = func_hi(theta_e[~idx])

    qei = NORM * np.square(ne) * fei
    return qei


def cooling_brems_ee(ne, te):
    """Cooling due to Bremsstrahlung from electron-electron scattering.

    NY95b Eq. 3.7
    """
    theta_e = K_BLTZ * te / MELC_C2

    def func_lo(_ne, th_e):
        NORM = 2.56e-22   # erg/cm^3/sec
        term = 1 + 1.1*th_e + np.square(th_e) - 1.25 * np.power(th_e, 2.5)
        return NORM * np.square(_ne) * np.power(th_e, 1.5) * term

    def func_hi(_ne, th_e):
        NORM = 3.4e-22   # erg/cm^3/sec
        term = np.log(1.123*th_e + 1.28)
        return NORM * np.square(_ne) * th_e * term

    if np.isscalar(theta_e):
        if theta_e <= 1.0:
            qei = func_lo(ne, theta_e)
        else:
            qei = func_hi(ne, theta_e)
    else:
        qei = np.zeros_like(theta_e)
        idx = (theta_e <= 1.0)
        qei[idx] = func_lo(ne[idx], theta_e[idx])
        qei[~idx] = func_hi(ne[~idx], theta_e[~idx])

    return qei
