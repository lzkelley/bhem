"""
"""

import os
import warnings

# Load Version Number from file (as string)
try:
    __VERSION_FNAME = "VERSION.txt"
    CWD = os.path.abspath(os.path.split(__file__)[0])
    vpath = os.path.join(CWD, os.pardir, __VERSION_FNAME)
    __version__ = open(vpath, 'r').read().strip()
except FileNotFoundError as err:
    warnings.warn("Error loading version file '{}'!  '{}'".format(vpath, str(err)), RuntimeWarning)
    __version__ = "v?.?.?"

PATH_DATA = os.path.join(CWD, "data", "")
if not os.path.exists(PATH_DATA):
    os.mkdir(PATH_DATA)

FAST_NRAD = 2000
FAST_RMIN = 3.0
FAST_RMAX = 1.0e5

TEMP_GRID_SIZE = 32

MASS_EXTR = [1e5, 5e10]
FEDD_EXTR = [1e-5, 1e-1]
RADS_EXTR = [3.0, 1e5]

from . import disks   # noqa
from . import spectra   # noqa
from . import fast_spectra  # noqa

# from .fast_spectra import FEDD_EXTR, MASS_EXTR, RADS_EXTR

from . constants import *   # noqa
