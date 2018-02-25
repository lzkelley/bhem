"""General Utility functions
"""

import numpy as np
import scipy as sp
import scipy.interpolate

from . import basics


def log_interp1d(xx, yy, kind='linear', **kwargs):
    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = sp.interpolate.interp1d(logx, logy, kind=kind, **kwargs)
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp


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
