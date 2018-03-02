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

from . import constants
from . import basics

from . constants import *
