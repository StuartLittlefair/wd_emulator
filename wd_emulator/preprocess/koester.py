# preprocess Koester grid into NumPy array of shape (n_teffs, n_logg, n_wavelengths)
import numpy as np
import glob
import os
from scipy.interpolate import interp1d
from astropy.utils.data import download_file


KOESTER_DA_URL = "https://github.com/StuartLittlefair/wd_emulator/model_grids/koester_da/"


def download_koester_grid():
    with open(download_file(KOESTER_DA_URL + "MANIFEST", pkgname='wd_emulator')) as manifest:
        filenames = manifest.readlines()

    all_files = []
    for filename in filenames:
        cache_name = download_file(
            KOESTER_DA_URL + filename.strip(),
            cache=True,
            pkgname='wd_emulator'
        )
        all_files.append(cache_name)
    return all_files


def read_file(path, log_g, teff):
    fn = f"da{int(teff):05d}_{int(100*log_g):03d}.dk.dat.txt"
    wave, flux = np.loadtxt(fn, unpack=True)
    return wave, flux


def preprocess_koester_grid(path, wl_range=[None, None], ranges=None):

    files = download_koester_grid()
    # unique Teffs and log_g, in ascending order
    filenames = [os.path.split(fn)[1] for fn in files]
    teffs = np.array(sorted(list(set(int(fn[2:7]) for fn in filenames))))
    gravities = np.array(sorted(list(set(float(fn[8:11])/100 for fn in filenames))))

    # Koester grid does not have truly identical wavelength scales, so cubic interpolate
    # onto a uniform wavelength scale. For this we pick one file to be the reference scale.
    wref, _ = read_file(path, 8.0, 20000)
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
            w, f = read_file(path, g, t)
            func = interp1d(w, f, kind='cubic', fill_value='extrapolate')
            spectra[it, ig] = func(wref)[index]
    return teffs, gravities, wref[index], spectra
