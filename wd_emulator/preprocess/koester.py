# preprocess Koester grid into NumPy array of shape (n_teffs, n_logg, n_wavelengths)
import numpy as np
import os
from scipy.interpolate import interp1d
from astropy.utils.data import download_file
from astropy.io import fits


KOESTER_DA_URL = "https://raw.githubusercontent.com/StuartLittlefair/wd_emulator/main/model_grids/koester_da/"


def download_koester_grid():
    return download_file(KOESTER_DA_URL + "grid.fits",
                         cache=True,
                         pkgname='wd_emulator')


def preprocess_koester_grid(wl_range=[None, None], ranges=None):
    """
    Pre-process the Koester DA grid ready for interpolation or emulation.

    The Koester model grid is not as nice as we'd like. Each individual model has a
    slightly different wavelength solution. The function below uses spline interpolation
    to put them all on an identical wavelength grid.

    At the same time we can restrict the wavelength range and/or restrict the grid to a range
    of temperatures and gravities.

    Parameters
    ----------
    wl_range: tuple, default (None, None)
        A pair of wavelength limits in Ansgtroms. Spectra will only be computed within this range.
        The default (None, None) returns the whole grid. A value of None for any limit will return
        the highest or lowest value in the grid.
    ranges: list or None
        A pair of Teff or log g ranges to apply to the grid. The default of None returns the whole
        grid.

        For example `ranges=[[20000, 40000], [7.5, 8.5]]` would restrict the grid to those
        temperature and log g ranges. Use None to specify no limit, so, for example,
        `ranges=[[20000, 40000], [None, None]]` would restrict the grid in temperature
        only.
    """
    grid_file = download_koester_grid()

    hdul = fits.open(grid_file)

    # unique Teffs and log_g, in ascending order
    teffs = np.array(sorted(list(set([hdu.header['teff'] for hdu in hdul[1:]]))))
    gravities = np.array(sorted(list(set([hdu.header['log_g'] for hdu in hdul[1:]]))))

    # Koester grid does not have truly identical wavelength scales, so cubic interpolate
    # onto a uniform wavelength scale. For this we pick one file to be the reference scale.
    wref = hdul[200].data['wavelength']
    wl_low = wref.min() if wl_range[0] is None else wl_range[0]
    wl_high = wref.max() if wl_range[1] is None else wl_range[1]
    index = (wref >= wl_low) & (wref <= wl_high)
    npix = np.sum(index)

    if ranges is not None:
        temp_range, grav_range = ranges
        temp_range[0] = teffs.min() if temp_range[0] is None else temp_range[0]
        temp_range[1] = teffs.max() if temp_range[1] is None else temp_range[1]
        grav_range[0] = gravities.min() if grav_range[0] is None else grav_range[0]
        grav_range[1] = gravities.max() if grav_range[1] is None else grav_range[1]
        teffs = teffs[(teffs >= temp_range[0]) & (teffs <= temp_range[1])]
        gravities = gravities[(gravities >= grav_range[0]) &
                              (gravities <= grav_range[1])]
    # make a container
    spectra = np.empty((teffs.size, gravities.size, npix))

    # read files into array
    for it, t in enumerate(teffs):
        for ig, g in enumerate(gravities):
            ext_name = f"{int(t)}_{int(100*g)}"
            ext = hdul[ext_name]
            w, f = ext.data['wavelength'], ext.data['flux']
            func = interp1d(w, f, kind='cubic', fill_value='extrapolate')
            spectra[it, ig] = func(wref)[index]
    return teffs, gravities, wref[index], spectra
