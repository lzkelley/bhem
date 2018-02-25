"""Radiation Calculations
"""

import numpy as np
import scipy as sp
import scipy.special

from . constants import SPLC, MELC, MPRT, K_BLTZ

CSQ = np.square(SPLC)
MELC_C2 = MELC * CSQ
MPRT_C2 = MPRT * CSQ


def coulomb_heating(ne, ni, te, ti):
    """Heating of electrons from Coulomb collisions with Ions.

    Based on NY95b Eq 3.1
    """

    NORM = 5.61e-32  # erg/cm^3/s

    kte = K_BLTZ * te
    kti = K_BLTZ * ti

    # Dimensionless temperatures (NY95b Eq. 3.2)
    theta_e = kte/MELC_C2
    theta_i = kti/MELC_C2
    te_ti = theta_e + theta_i
    red_temp = te_ti / (theta_e * theta_i)

    k2e = sp.special.kn(2, 1/theta_e)
    k2i = sp.special.kn(2, 1/theta_i)
    k1r = sp.special.kn(1, red_temp)
    k0r = sp.special.kn(0, red_temp)

    term1 = ne * ni * (ti - te) / (k2e * k2i)
    term2 = (2*np.square(te_ti) + 1) * k1r / te_ti
    term3 = 2*k0r

    qie = NORM * term1 * (term2 + term3)
    return qie
