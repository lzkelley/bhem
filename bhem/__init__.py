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

MASS_EXTR = [5e5, 1e11]
FEDD_EXTR = [1e-5, 1e-1]
RADS_EXTR = [2.99, 1.01e5]

from . import disks   # noqa
from . import spectra   # noqa
from . import fast_spectra  # noqa
from . import obs   # noqa
from . import bel   # noqa

from .fast_spectra import FEDD_EXTR, MASS_EXTR, RADS_EXTR

from . constants import *   # noqa
