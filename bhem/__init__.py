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

from . import disks   # noqa
from . import spectra   # noqa
from . import fast_spectra  # noqa

from . constants import *   # noqa
