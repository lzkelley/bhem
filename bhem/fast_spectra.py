"""
"""
import datetime
import os
# import sys
import logging

import numpy as np
import scipy as sp
import scipy.optimize  # noqa
# import tqdm
import h5py

import zcode.inout as zio
import zcode.math as zmath

from . import spectra, radiation  # , utils
from . import PATH_DATA, TEMP_GRID_SIZE, MASS_EXTR, FEDD_EXTR, RADS_EXTR
from . constants import MSOL, MELC, MPRT, SPLC, K_BLTZ, H_PLNK

# TEMP_GRID_SIZE = 32
#
# MASS_EXTR = [1e6, 5e10]
# FEDD_EXTR = [1e-5, 1e-1]
# RADS_EXTR = [3.0, 1e5]

GRID_NAMES = ['mass', 'fedd', 'rmin', 'rmax']

ALPHA_VISC = 0.1
BETA_GP = 0.5
FRAC_ADV = 0.5

GAMMA_SH = (32 - 24*BETA_GP - 3*BETA_GP**2) / (24 - 21*BETA_GP)
EPS = (5/3 - GAMMA_SH) / (GAMMA_SH - 1.0)
EPS_PRIME = EPS / FRAC_ADV

DELTA = MELC/MPRT
GAE = np.sqrt(1.0 + 18.0 * np.square(ALPHA_VISC/(5.0 + 2*EPS_PRIME))) - 1.0
C1 = GAE * (5 + 2*EPS_PRIME) / (3 * np.square(ALPHA_VISC))
# C2 = np.sqrt(2 * EPS_PRIME * C1 / 3)
C3 = 2 * C1 / 3
MEC2 = MELC * SPLC**2
S1 = 1.42e9 * np.sqrt(1 - BETA_GP) * np.sqrt(C3 / C1 / ALPHA_VISC)
S3 = 1.05e-24

KB_OVER_MEC2 = K_BLTZ / MEC2

META = dict(ALPHA_VISC=ALPHA_VISC, BETA_GP=BETA_GP, FRAC_ADV=FRAC_ADV)


def main(num=None, recreate=True):
    if num is None:
        num = TEMP_GRID_SIZE

    fname = grid_fname(num)
    exists = os.path.exists(fname)
    logging.warning("Grid for num={} exists: {} ({})".format(num, exists, fname))
    logging.info("recreate: {}".format(recreate))
    if not exists or recreate:
        grid, grid_names, grid_temps, grid_valid = get_temp_grid(num)
        save_grid(fname, grid, grid_names, grid_temps, grid_valid)

    return


def get_interp(num=None):
    if num is None:
        num = TEMP_GRID_SIZE
    fname = grid_fname(num)
    grid, grid_names, grid_temps, grid_valid = load_grid(fname=fname)

    grid_temps[~grid_valid] = np.mean(grid_temps[grid_valid])

    # mesh = np.meshgrid(*grid)
    # mesh = np.log10(mesh)
    mesh = [np.log10(gg) for gg in grid]
    grid_temps = np.log10(grid_temps)
    interp_ll = sp.interpolate.RegularGridInterpolator(mesh, grid_temps)

    def interp(xx):
        try:
            res = 10**interp_ll(np.log10(xx))
        except ValueError:
            for ii, gg in enumerate(interp_ll.grid):
                logging.error("\tparam: '{}'".format(ii))
                logging.error("\t\tgrid: {}".format(zmath.minmax(gg)))
                logging.error("\t\targ : {}".format(zmath.minmax(np.log10(xx)[:, ii])))
            logging.error("ValueError for argument: '{}'".format(xx))
            logging.error("ValueError for argument: log: '{}'".format(np.log10(xx)))
            raise
        return res

    return interp


def grid_fname(num):
    fname = "temp_grid_n{}.hdf5".format(num)
    fname = os.path.join(PATH_DATA, fname)
    return fname


def save_grid(fname, grid, grid_names, grid_temps, grid_valid):
    fname = os.path.abspath(fname)
    with h5py.File(fname, 'w') as out:
        group = out.create_group('grid')
        for nn, vv in zip(grid_names, grid):
            group.create_dataset(nn, data=vv)

        group = out.create_group('parameters')
        for nn, vv in META.items():
            group.create_dataset(nn, data=vv)

        out.create_dataset('temps', data=grid_temps)
        out.create_dataset('valid', data=grid_valid)

    logging.info("Saved to '{}' size '{}'".format(fname, zio.get_file_size(fname)))
    return


def load_grid(*args, num=None, fname=None):
    if len(args):
        raise ValueError("Only passed kwargs to `load_grid()`!")
    if fname is None:
        if num is None:
            num = TEMP_GRID_SIZE
        fname = grid_fname(num)

    fname = os.path.abspath(fname)
    if not os.path.exists(fname):
        raise ValueError("fname '{}' does not exist!".format(fname))
    with h5py.File(fname, 'r') as h5:
        grid_group = h5['grid']
        # grid_names = list(grid_group.keys())
        grid_names = []
        grid = []
        for nn in GRID_NAMES:
            grid.append(grid_group[nn][:])
            grid_names.append(nn)

        grid_temps = h5['temps'][:]
        grid_valid = h5['valid'][:]

    return grid, grid_names, grid_temps, grid_valid


def get_temp_grid(num, fix=True):
    grid_extr = [np.array(MASS_EXTR)*MSOL, FEDD_EXTR, RADS_EXTR, RADS_EXTR]
    grid_names = ['mass', 'fedd', 'rmin', 'rmax']
    grid = [np.logspace(*np.log10(extr), num) for extr in grid_extr]
    shape = [num for ii in range(len(grid))]
    tot = np.product(shape)
    grid_temps = np.zeros(shape)
    grid_valid = np.ones(shape, dtype=bool)

    cnt = 0
    beg = datetime.datetime.now()
    # for idx in tqdm.tqdm_notebook(np.ndindex(*shape), total=tot):
    for idx in zio.tqdm(np.ndindex(*shape), total=tot):
        # print(idx)
        vals = [gg[ii] for gg, ii in zip(grid, idx)]
        if vals[2] >= vals[3]:
            grid_valid[idx] = False
            continue
        tt = solve_adaf_temp(*vals)
        if tt is not None:
            grid_temps[idx] = tt
        cnt += 1

    end = datetime.datetime.now()
    dur = (end - beg)
    dur_per = dur.total_seconds()/cnt
    bads_nan = np.isnan(grid_temps)
    grid_temps = np.nan_to_num(grid_temps)
    bads = grid_valid & np.isclose(grid_temps, 0.0)

    logging.warning("Success on : {}".format(zmath.frac_str(grid_temps[grid_valid] > 0.0)))
    logging.warning("nan values: {}".format(zmath.frac_str(bads_nan)))
    logging.warning("Bad values: {}".format(zmath.frac_str(bads)))
    logging.warning("Done after {}, per iteration: {}".format(str(dur), dur_per))

    if fix:
        grid_temps = interp_bad_grid_vals(grid, grid_temps, grid_valid)

    return grid, grid_names, grid_temps, grid_valid


def solve_adaf_temp(mass, fedd, rmin, rmax, debug=False):

    msol = mass / MSOL
    lvl = logging.WARNING

    def heat_cool(temp):
        """Calculate heating and cooling rates for disk as a whole.
        """

        nonlocal mass, fedd, rmin, rmax, msol

        alpha = ALPHA_VISC
        beta = BETA_GP
        eps_prime = EPS_PRIME

        delta = DELTA
        rmin = rmin
        rmax = rmax

        theta_e = KB_OVER_MEC2 * temp
        xm = spectra.xm_from_te(temp, msol, fedd)

        tau_es = 23.87 * fedd * (0.3 / alpha) * (0.5 / C1) * np.sqrt(3/rmin)
        mean_amp_a = 1.0 + 4.0 * theta_e + 16*np.square(theta_e)
        alpha_crit = - np.log(tau_es) / np.log(mean_amp_a)

        s2 = 1.19e-13 * xm

        # Viscous Heating
        # ---------------
        _ge = radiation._heat_func_g(theta_e)
        q1 = 1.2e38 * _ge * C3 * beta * msol * np.square(fedd) / np.square(alpha*C1) / rmin
        q2 = delta * 9.39e38 * eps_prime * C3 * msol * fedd / rmin
        heat_elc = q1 + q2

        # Synchrotron
        # -----------
        # Eq. 24  [Hz]
        f_p = S1 * s2 * np.sqrt(fedd/msol) * np.square(temp) * np.power(rmin, -1.25)
        lum_synch_peak = np.power(S1 * s2, 3) * S3 * np.power(rmin, -1.75) * np.sqrt(msol)
        lum_synch_peak *= np.power(fedd, 1.5) * np.power(temp, 7) / f_p

        # Eq. 26
        power_synch = 5.3e35 * np.power(xm/1000, 3) * np.power(alpha/0.3, -1.5)
        power_synch *= np.power((1 - beta)/0.5, 1.5) * np.power(C1/0.5, -1.5)

        # Bremsstrahlung
        # --------------
        # Eq. 29
        power_brems = 4.78e34 * np.log(rmax/rmin) / np.square(alpha * C1)
        power_brems *= radiation._brems_fit_func_f(theta_e) * fedd * msol

        # Compton
        # -------
        power_compt = lum_synch_peak * f_p / (1 - alpha_crit)
        power_compt *= (np.power(6.2e7 * (temp/1e9) / (f_p/1e12), 1 - alpha_crit) - 1.0)

        return heat_elc, power_synch, power_brems, power_compt

    def _func(logt):
        tt = np.power(10.0, logt)
        qv, qs, qb, qc = heat_cool(tt)
        rv = qv - (qs + qb + qc)
        return rv

    start_temps = [1e11, 1e10, 1e12, 1e9, 1e8]
    success = False
    for ii, t0 in enumerate(start_temps):
        try:
            logt = sp.optimize.newton(_func, np.log10(t0), tol=1e-4, maxiter=100)
            temp_e = np.power(10.0, logt)
        except (RuntimeError, FloatingPointError) as err:
            if debug:
                logging.warn("Trial '{}' (t={:.1e}) optimization failed: {}".format(
                    ii, t0, str(err)))
        else:
            success = True
            break

    if success:
        # logging.log(lvl, "Success with `t0`={:.2e} ==> t={:.2e}".format(t0, temp_e))
        pass
    else:
        err = ("Unable to find electron temperature!"
               "\nIf the eddington factor is larger than 1e-2, "
               "this may be expected!")
        if debug:
            logging.log(lvl, "FAILED to find electron temperature!")
            logging.log(lvl, "m = {:.2e}, f = {:.2e}".format(msol, fedd))
            logging.log(lvl, err)
        # raise RuntimeError(err)
        return None

    qv, qs, qb, qc = heat_cool(temp_e)
    heat = qv
    cool = qs + qb + qc
    diff = np.fabs(heat - cool) / heat

    if diff < 1e-2:
        if debug:
            logging.log(lvl, "Heating vs. cooling frac-diff: {:.2e}".format(diff))
    else:
        if debug:
            err = "Electron temperature seems inconsistent (Te = {:.2e})!".format(temp_e)
            err += "\n\tm: {:.2e}, f: {:.2e}".format(msol, fedd)
            err += "\n\tHeating: {:.2e}, Cooling: {:.2e}, diff: {:.4e}".format(heat, cool, diff)
            err += "\n\tThis may mean there is an input error (e.g. mdot may be too large... or small?)."
            logging.log(lvl, err)
        return None

    return temp_e


def interp_bad_grid_vals(grid, grid_temps, grid_valid):
    grid_temps = np.copy(grid_temps)
    bads = grid_valid & np.isclose(grid_temps, 0.0)
    shape = [len(gg) for gg in grid]
    logging.warning("Fixing bad values: {}".format(zmath.frac_str(bads)))

    neighbors = []
    good_neighbors = []
    bads_inds = np.array(np.where(bads)).T
    # for bad in tqdm.tqdm_notebook(bads_inds):
    for bad in zio.tqdm(bads_inds):
        nbs = []

        # print(bad)
        cnt = 0
        for dim in range(4):
            for side in [-1, +1]:
                test = [bb for bb in bad]
                test[dim] += side
                if test[dim] < 0 or test[dim] >= shape[dim]:
                    continue
                test = tuple(test)
                # print("\t", test)
                # print("\t", temps[test])
                nbs.append(test)
                if grid_temps[test] > 0.0:
                    cnt += 1
        neighbors.append(nbs)
        good_neighbors.append(cnt)

    num_nbs = [len(nbs) for nbs in neighbors]
    logging.warning("All  neighbors: {}".format(zmath.stats_str(num_nbs)))
    logging.warning("Good neighbors: {}".format(zmath.stats_str(good_neighbors)))
    goods = np.zeros(len(neighbors))

    MAX_TRIES = 10
    still_bad = list(np.argsort(good_neighbors)[::-1])
    tries = 0
    while len(still_bad) > 0 and tries < MAX_TRIES:
        keep_bad = []
        for kk, ii in enumerate(still_bad):
            values = np.zeros(num_nbs[ii])
            for jj, nbr in enumerate(neighbors[ii]):
                values[jj] = grid_temps[nbr]

            cnt = np.count_nonzero(values)
            if cnt == 0:
                keep_bad.append(kk)
                continue

            new = np.sum(np.log10(values[values > 0])) / cnt
            loc = tuple(bads_inds[ii])
            # print("\t", loc, new, cnt)
            grid_temps[loc] = 10**new
            goods[ii] = cnt

        still_bad = [still_bad[kk] for kk in keep_bad]
        num_still = len(still_bad)
        logging.warning("Try: {}, still_bad: {}".format(tries, num_still))
        if (tries+1 >= MAX_TRIES) and (num_still > 0):
            logging.error("After {} tries, still {} bad!!".format(tries, num_still))

        tries += 1

    logging.warning("Filled neighbors: {}".format(zmath.stats_str(goods)))
    logging.warning("Full temps array: {}".format(zmath.stats_str(grid_temps[grid_valid])))
    return grid_temps


def plot_grid(grid, grid_names, temps, valid, interp=None):
    import matplotlib.pyplot as plt
    import zcode.plot as zplot

    extr = zmath.minmax(temps, filter='>')
    smap = zplot.colormap(extr, 'viridis')

    # bads = valid & np.isclose(temps, 0.0)

    num = len(grid)
    fig, axes = plt.subplots(figsize=[14, 14], nrows=num, ncols=num)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    def_idx = [-4, -4, 4, -4]

    for (ii, jj), ax in np.ndenumerate(axes):
        if ii < jj:
            ax.set_visible(False)
            continue

        ax.set(xscale='log', yscale='log')
        xx = grid[jj]
        if ii == jj:
            # print(grid_names[ii], zmath.minmax(grid[ii], filter='>'))
            # idx = list(range(num))
            # idx.pop(ii)
            # idx = tuple(idx)
            # vals = np.mean(temps, axis=idx)

            idx = [slice(None) if aa == ii else def_idx[aa] for aa in range(num)]
            vals = temps[tuple(idx)]
            ax.plot(xx, vals, 'k-')

            if interp is not None:
                num_test = 10
                test = [np.ones(num_test)*grid[aa][def_idx[aa]] for aa in range(num)]
                test[ii] = zmath.spacing(grid[ii], 'log', num_test)
                test_vals = [interp(tt) for tt in np.array(test).T]
                ax.plot(test[ii], test_vals, 'r--')

            # bad_vals = np.count_nonzero(bads, axis=idx)
            # tw = ax.twinx()
            # tw.plot(xx, bad_vals, 'r--')

        else:
            # print(ii, jj)
            # print("\t", ii, grid_names[ii], zmath.minmax(grid[ii], filter='>'))
            # print("\t", jj, grid_names[jj], zmath.minmax(grid[jj], filter='>'))
            # idx = [0, 1, 2, 3]
            # idx.pop(np.max([ii, jj]))
            # idx.pop(np.min([ii, jj]))
            # vals = np.mean(temps, axis=tuple(idx))

            # idx = [slice(None) if aa in [ii, jj] else num//2 for aa in range(num)]
            idx = [slice(None) if aa in [ii, jj] else def_idx[aa] for aa in range(num)]
            vals = temps[tuple(idx)]
            if len(vals) == 0:
                continue

            yy = grid[ii]
            xx, yy = np.meshgrid(xx, yy, indexing='ij')
            ax.pcolor(xx, yy, vals, cmap=smap.cmap, norm=smap.norm)

            if np.count_nonzero(vals > 0.0) == 0:
                continue

            tit = "{:.1e}, {:.1e}".format(*zmath.minmax(vals, filter='>'))
            ax.set_title(tit, size=10)

            # bad_vals = np.count_nonzero(bads, axis=tuple(idx))
            # idx = (bad_vals > 0.0)
            # aa = xx[idx]
            # bb = yy[idx]
            # cc = bad_vals[idx]
            # ax.scatter(aa, bb, s=2*cc**2, color='0.5', alpha=0.5)
            # ax.scatter(aa, bb, s=cc**2, color='r')

            if interp is not None:
                for kk in range(10):
                    idx = (vals > 0.0)
                    x0 = 10**np.random.uniform(*zmath.minmax(np.log10(xx[idx])))
                    y0 = 10**np.random.uniform(*zmath.minmax(np.log10(yy[idx])))
                    # y0 = np.random.choice(yy[idx])

                    temp = [grid[ll][def_idx[ll]] for ll in range(num)]
                    temp[ii] = y0
                    temp[jj] = x0

                    if temp[2] >= temp[3]:
                        temp[2] = 3.1
                    iv = interp(temp)
                    if not np.isfinite(iv) or np.isclose(iv, 0.0):
                        print("\nBAD")
                        print(temp)
                        print(iv)

                        for kk in range(num):
                            if def_idx[kk] == 0:
                                temp[kk] = temp[kk] * 1.11
                            elif def_idx[kk] == -1:
                                temp[kk] = 0.99 * temp[kk]

                        iv = interp(temp)
                        print("\t", temp)
                        print("\t", iv)

                    cc = smap.to_rgba(iv)
                    ss = 20
                    ax.scatter(temp[jj], temp[ii], color='0.5', s=2*ss)
                    ax.scatter(temp[jj], temp[ii], color=cc, s=ss)

        if ii == num-1:
            ax.set_xlabel(grid_names[jj])
        if jj == 0 and ii != 0:
            ax.set_ylabel(grid_names[ii])

    return fig


class Fast_Mahadevan96:

    def __init__(self, mass, fedd, rmin, rmax, temp_e=None, interp=None):
        """
        """
        self.mass = mass
        # Mass in units of solar=masses
        self.msol = mass/MSOL
        self.fedd = fedd

        self.rmin = rmin
        self.rmax = rmax

        if temp_e is None:
            if interp is None:
                interp = get_interp()

            temp_e = interp([mass, fedd, rmin, rmax])

        self.temp_e = temp_e
        xm_e = spectra.xm_from_te(temp_e, self.msol, fedd)
        self.s2 = 1.19e-13 * xm_e

        theta_e = radiation.dimensionless_temperature_theta(temp_e, MELC)
        # Eq. 31
        tau_es = 23.87 * fedd * (0.3 / ALPHA_VISC) * (0.5 / C1) * np.sqrt(3/rmin)
        # Eq. 32
        mean_amp_a = 1.0 + 4.0 * theta_e + 16*np.square(theta_e)
        # Eq. 34
        self.alpha_crit = - np.log(tau_es) / np.log(mean_amp_a)
        return

    def spectrum(self, freqs):
        synch = self._calc_spectrum_synch(freqs)
        brems = self._calc_spectrum_brems(freqs)
        compt = self._calc_spectrum_compt(freqs)

        spectrum = synch + brems + compt
        return spectrum

    def _calc_spectrum_synch(self, freqs):
        """Mahadevan 1996 - Eq. 25

        Cutoff above peak frequency (i.e. ignore exponential portion).
        Ignore low-frequency transition to steeper (22/13 slope) from rmax.
        """
        msol = self.msol
        fedd = self.fedd

        scalar = np.isscalar(freqs)
        freqs = np.atleast_1d(freqs)

        lnu = S3 * np.power(S1*self.s2, 1.6)
        lnu *= np.power(msol, 1.2) * np.power(fedd, 0.8)
        lnu *= np.power(self.temp_e, 4.2) * np.power(freqs, 0.4)

        nu_p = self._freq_synch_peak(self.temp_e, msol, fedd)
        lnu[freqs > nu_p] = 0.0
        if scalar:
            lnu = np.squeeze(lnu)

        return lnu

    def _calc_spectrum_brems(self, freqs):
        """Mahadevan 1996 - Eq. 30
        """
        msol = self.msol
        fedd = self.fedd
        temp = self.temp_e
        const = 2.29e24   # erg/s/Hz

        scalar = np.isscalar(freqs)
        freqs = np.atleast_1d(freqs)

        t1 = np.log(self.rmax/self.rmin) / np.square(ALPHA_VISC * C1)
        t2 = np.exp(-H_PLNK*freqs / (K_BLTZ * temp)) * msol * np.square(fedd) / temp
        fe = radiation._brems_fit_func_f(temp)
        lbrems = const * t1 * fe * t2
        if scalar:
            lbrems = np.squeeze(lbrems)

        return lbrems

    def _calc_spectrum_compt(self, freqs):
        """Compton Scattering spectrum from upscattering of Synchrotron photons.

        Mahadevan 1996 - Eq. 38
        """
        fedd = self.fedd
        temp = self.temp_e

        scalar = np.isscalar(freqs)
        freqs = np.atleast_1d(freqs)

        f_p, l_p = self._synch_peak(fedd, self.msol, temp)
        lsp = np.power(freqs/f_p, -self.alpha_crit) * l_p
        lsp[freqs < f_p] = 0.0

        # See Eq. 35
        max_freq = 3*K_BLTZ*temp/H_PLNK
        lsp[freqs > max_freq] = 0.0
        if scalar:
            lsp = np.squeeze(lsp)

        return lsp

    def _freq_synch_peak(self, temp, msol, fedd):
        """Mahadevan 1996 Eq. 24
        """
        nu_p = S1 * self.s2 * np.sqrt(fedd/msol) * np.square(temp) * np.power(self.rmin, -1.25)
        return nu_p

    def _synch_peak(self, fedd, msol, temp):
        f_p = self._freq_synch_peak(temp, msol, fedd)
        l_p = np.power(S1 * self.s2, 3) * S3 * np.power(self.rmin, -1.75) * np.sqrt(msol)
        l_p *= np.power(fedd, 1.5) * np.power(temp, 7) / f_p
        return f_p, l_p


class Fast_Mahadevan96_Array:

    def __init__(self, mass, fedd, rmin, rmax, temp_e=None, interp=None):
        """
        """
        self.mass = mass
        # Mass in units of solar=masses
        self.msol = mass/MSOL
        self.fedd = fedd

        self.rmin = rmin
        self.rmax = rmax

        if temp_e is None:
            if interp is None:
                interp = get_interp()

            args = [mass, fedd, rmin, rmax]
            shp = np.shape(args[0])
            if not np.all([shp == np.shape(aa) for aa in args]):
                all_shps = [np.shape(aa) for aa in args]
                print("all shapes = ", all_shps)
                raise ValueError("Shape mismatch!")
            args = [aa.flatten() for aa in args]
            args = np.array(args).T
            temp_e = interp(args)
            temp_e = temp_e.reshape(shp)
            assert np.shape(temp_e) == np.shape(mass), "Output shape mismatch!"

        self.temp_e = temp_e
        xm_e = spectra.xm_from_te(temp_e, self.msol, fedd)
        self.s2 = 1.19e-13 * xm_e

        theta_e = radiation.dimensionless_temperature_theta(temp_e, MELC)
        # Eq. 31
        tau_es = 23.87 * fedd * (0.3 / ALPHA_VISC) * (0.5 / C1) * np.sqrt(3/rmin)
        # Eq. 32
        mean_amp_a = 1.0 + 4.0 * theta_e + 16*np.square(theta_e)
        # Eq. 34
        self.alpha_crit = - np.log(tau_es) / np.log(mean_amp_a)
        return

    def spectrum(self, freqs):
        synch = self._calc_spectrum_synch(freqs)
        brems = self._calc_spectrum_brems(freqs)
        compt = self._calc_spectrum_compt(freqs)

        spectrum = synch + brems + compt
        return spectrum

    def _calc_spectrum_synch(self, freqs):
        """Mahadevan 1996 - Eq. 25

        Cutoff above peak frequency (i.e. ignore exponential portion).
        Ignore low-frequency transition to steeper (22/13 slope) from rmax.
        """
        msol = self.msol
        fedd = self.fedd

        scalar = np.isscalar(freqs)
        freqs = np.atleast_1d(freqs)

        lnu = S3 * np.power(S1*self.s2, 1.6)
        # lnu *= np.power(msol, 1.2) * np.power(fedd, 0.8)
        # lnu *= np.power(self.temp_e, 4.2) * np.power(freqs, 0.4)
        lnu = lnu * np.power(msol, 1.2) * np.power(fedd, 0.8)
        lnu = lnu * np.power(self.temp_e, 4.2) * np.power(freqs, 0.4)

        nu_p = self._freq_synch_peak(self.temp_e, msol, fedd)
        lnu[freqs > nu_p] = 0.0
        if scalar:
            lnu = np.squeeze(lnu)

        return lnu

    def _calc_spectrum_brems(self, freqs):
        """Mahadevan 1996 - Eq. 30
        """
        msol = self.msol
        fedd = self.fedd
        temp = self.temp_e
        const = 2.29e24   # erg/s/Hz

        scalar = np.isscalar(freqs)
        freqs = np.atleast_1d(freqs)

        t1 = np.log(self.rmax/self.rmin) / np.square(ALPHA_VISC * C1)
        t2 = np.exp(-H_PLNK*freqs / (K_BLTZ * temp)) * msol * np.square(fedd) / temp
        fe = radiation._brems_fit_func_f(temp)
        lbrems = const * t1 * fe * t2
        if scalar:
            lbrems = np.squeeze(lbrems)

        return lbrems

    def _calc_spectrum_compt(self, freqs):
        """Compton Scattering spectrum from upscattering of Synchrotron photons.

        Mahadevan 1996 - Eq. 38
        """
        fedd = self.fedd
        temp = self.temp_e

        scalar = np.isscalar(freqs)
        freqs = np.atleast_1d(freqs)

        f_p, l_p = self._synch_peak(fedd, self.msol, temp)
        lsp = np.power(freqs/f_p, -self.alpha_crit) * l_p
        lsp[freqs < f_p] = 0.0

        # See Eq. 35
        max_freq = 3*K_BLTZ*temp/H_PLNK
        lsp[freqs > max_freq] = 0.0
        if scalar:
            lsp = np.squeeze(lsp)

        return lsp

    def _freq_synch_peak(self, temp, msol, fedd):
        """Mahadevan 1996 Eq. 24
        """
        nu_p = S1 * self.s2 * np.sqrt(fedd/msol) * np.square(temp) * np.power(self.rmin, -1.25)
        return nu_p

    def _synch_peak(self, fedd, msol, temp):
        f_p = self._freq_synch_peak(temp, msol, fedd)
        l_p = np.power(S1 * self.s2, 3) * S3 * np.power(self.rmin, -1.75) * np.sqrt(msol)
        l_p *= np.power(fedd, 1.5) * np.power(temp, 7) / f_p
        return f_p, l_p


if __name__ == "__main__":
    np.seterr(divide='ignore', invalid='ignore', over='ignore')
    main()
