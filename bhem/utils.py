"""General Utility functions
"""

import numpy as np
import scipy as sp
import scipy.interpolate

from . import basics
from . constants import MPRT


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
