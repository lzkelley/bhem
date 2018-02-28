"""Radiation Calculations
"""

import numpy as np
import scipy as sp
import scipy.special  # noqa

from . constants import SPLC, MELC, MPRT, K_BLTZ, QELC, H_PLNK

CSQ = np.square(SPLC)
MELC_C2 = MELC * CSQ
MPRT_C2 = MPRT * CSQ


def _heat_func_g(th_e):
    t1 = 1/sp.special.kn(2, 1/th_e)
    t2 = 2 + 2*th_e + 1/th_e
    t3 = np.exp(-1/th_e)
    return t1*t2*t3


def heating_coulomb_ie(ne, ni, te, ti):
    """Heating of electrons from Coulomb collisions with Ions.

    Based on NY95b Eq 3.1

    The Bessel functions (`K_n(x)`) used in this calculated approach zero *very quickly*
    for `x >> 1`.  This causes numerical issues when using the expression as given (Eq. 3.1).
    Instead, in problematic regimes, use approximation for large x, i.e. that

        lim x->inf  K_n(x) = a * exp(-x) / sqrt(x),   where  a = sqrt(pi/2)

    This approach seems to be accurate to at worst about 10%.

    """

    NORM = 5.61e-32  # erg/cm^3/s

    # where to use approximations for Bessel functions
    #    Note that the arguments to bessel functions are ~ 1/theta (the dimensionless temperatures)
    CUTOFF = 1.0/30.0

    scalar = np.isscalar(te) and np.isscalar(ti)
    if scalar:
        te = np.atleast_1d(te)
        ti = np.atleast_1d(ti)
    elif np.isscalar(te):
        te = te * np.ones_like(ti)
    elif np.isscalar(ti):
        ti = ti * np.ones_like(te)

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

    # Convert back to scalar
    if scalar:
        assert np.size(qie) == 1, "Heating should be scalar!"
        qie = qie[0]

    return qie


def _brems_fit_func_f(te):
    """Bremsstrahlung emission fitting function F(theta_e)

    NY95b - Eq 3.6
    """
    theta_e = K_BLTZ * te / MELC_C2

    def func_lo(th_e):
        return 4 * np.sqrt(2*th_e/np.power(np.pi, 3)) * (1 + 1.781 * np.power(th_e, 1.34))

    def func_hi(th_e):
        return 9 * th_e * (np.log(1.123*th_e + 0.48) + 1.5) / (2*np.pi)

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

    return fei


def cooling_brems_ei(ne, te):
    """Cooling due to Bremsstrahlung from electron-ion scattering.

    NY95b Eq. 3.5
    """
    NORM = 1.48e-22   # erg/cm^3/sec
    fei = _brems_fit_func_f(te)
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


def dimensionless_temperature_theta(temp, mass=MELC):
    theta = K_BLTZ * temp / (mass * SPLC * SPLC)
    return theta


def _synch_fit_func_iprime(xm):
    """
    NY95b Eq. 3.12
    """
    xm2 = np.power(xm, 0.5)
    xm3 = np.power(xm, 1/3)
    xm4 = np.sqrt(xm2)
    xm6 = np.sqrt(xm3)
    t1 = 4.0505/xm6
    t2 = 1.0 + 0.4/xm4 + 0.5316/xm2
    t3 = np.exp(-1.8899*xm3)
    ip = t1 * t2 * t3
    return ip


def synchrotron_thin_spectrum(freqs, ne, te, bfield):
    """Optically thin (unobsorbed) synchrotron spectrum.

    Units of erg/cm^3/s/Hz

    NY95b Eq 3.9
    """
    const = 4.43e-30   # erg/cm^3/s/Hz
    theta_e = K_BLTZ * te / (MELC * SPLC * SPLC)
    v0 = QELC * bfield / (2*np.pi*MELC*SPLC)
    xm = 2*freqs/(3*v0*np.square(theta_e))
    iprime = _synch_fit_func_iprime(xm)
    esyn = const * 4*np.pi*ne*freqs*iprime/sp.special.kn(2, 1/theta_e)
    return esyn


def black_body_spectrum(freqs, te):
    """Planck's Law Black-Body spectral density.
    """
    denom = np.exp(H_PLNK*freqs/K_BLTZ/te) - 1
    bv = 2*H_PLNK*np.power(freqs, 3) / (SPLC*SPLC * denom)
    return bv
