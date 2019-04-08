"""General Utility functions
"""

import sys

import numpy as np
import scipy as sp
import scipy.interpolate  # noqa

from . import basics
from . constants import MPRT


def log_interp1d(xx, yy, kind='linear', **kwargs):
    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = sp.interpolate.interp1d(logx, logy, kind=kind, **kwargs)
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp


def log_midpoints(arr):
    mid = 0.5 * (np.log10(arr[1:]) + np.log10(arr[:-1]))
    mid = np.power(10.0, mid)
    return mid


def mdot_fedd(mass, mdot, fedd, raise_error=True):
    """Make sure that `mdot` and `fedd` are set.

    Returns
    -------
    mdot
    fedd

    """
    mdot_edd = basics.eddington_accretion(mass)

    if mdot is not None:
        fedd = mdot / mdot_edd
        return mdot, fedd

    if fedd is not None:
        mdot = fedd * mdot_edd
        return mdot, fedd

    if raise_error:
        raise ValueError("`mdot` or `fedd` must be given!")

    return None, None


def ndens_elc(dens, frac_hmass):
    """Calculate the number-density of electrons based on the mass-density and hydrogen mass frac.
    """
    X = frac_hmass   # Hydrogen mass-frac
    Y = 1.0 - X      # Helium mass-fraction
    Z = 0.0          # Higher-element mass-fraction
    nde = (dens/MPRT) * (X + 2*Y/4 + Z/2)
    return nde


def ndens_ion(dens, frac_hmass):
    """Calculate the number-density of ions based on the mass-density and hydrogen mass frac.
    """
    X = frac_hmass   # Hydrogen mass-frac
    Y = 1.0 - X      # Helium mass-fraction
    Z = 0.0          # Higher-element mass-fraction
    # This assumes an average charge for heavier-elements of ~8
    nde = (dens/MPRT) * (X + Y/4 + Z/16)
    return nde


def get_log(name='bhem', level=30, stream=sys.stdout):
    import logging

    log = logging.getLogger(name)
    # Make sure handlers don't get duplicated (ipython issue)
    while len(log.handlers) > 0:
        log.handlers.pop()
    # Prevents duplication or something something...
    log.propagate = 0
    log.setLevel(level)

    format_date = '%Y/%m/%d %H:%M:%S'
    format_stream = "%(message)s"
    formatter = logging.Formatter(format_stream, format_date)

    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)
    handler.setLevel(level)

    log.addHandler(handler)

    return log
