"""Constant parameters.
"""

import numpy as np
import scipy as sp
import scipy.interpolate

from collections import namedtuple

# Stefan-Boltzmann constant
SIGMA_SB = 5.6703669e-05      # gram / kelvin^4 / sec^3

# Thomson-scattering cross-section
SIGMA_T = 6.6524587158e-25    # cm^2

# Newton's gravitational constant
NWTG = 6.674079e-08           # cm^3 / gram  / sec^2

# Solar Mass
MSOL = 1.988475e+33           # gram

# Speed of light
SPLC = 2.997924e10            # cm / s

# Proton Mass
MPRT = 1.672621898e-24        # gram

# Planck's constant
H_PLNK = 6.62607004e-27       # erg s

# Boltzmann Constant
K_BLTZ = 1.38064852e-16       # erg / K

_band_name = ['u', 'b', 'v', 'r', 'i']
_band_wlen = [365, 445, 551, 658, 806]   # nm
_band_color = ['violet', 'blue', 'green', 'red', 'darkred']
Band = namedtuple('band', ['name', 'freq', 'wlen', 'color'])

BANDS = {nn: Band(nn, SPLC/(ll*1e-7), ll*1e-7, cc)
         for nn, ll, cc in zip(_band_name, _band_wlen, _band_color)}


def log_interp1d(xx, yy, kind='linear', **kwargs):
    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = sp.interpolate.interp1d(logx, logy, kind=kind, **kwargs)
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp
