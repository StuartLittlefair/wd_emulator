from Starfish.grid_tools import GridInterface
from Starfish import constants as C
import numpy as np
import glob
import os
from scipy.interpolate import InterpolatedUnivariateSpline


class KoesterGridInterface(GridInterface):

    def __init__(self, path, wl_range=None):
        files = glob.glob(os.path.join(path, 'da*.txt'))
        filenames = [os.path.split(fn)[1] for fn in files]
        teffs = np.array(sorted(list(set(int(fn[2:7]) for fn in filenames))))
        gravities = np.array(sorted(list(set(float(fn[8:11])/100 for fn in filenames))))
        param_names = ['teff', 'logg']
        points = [teffs, gravities]
        self.path = path
        self.wref, _ = np.loadtxt(self._filename(20000, 8.0), unpack=True)
        if wl_range is None:
            wl_range = (self.wref.min(), self.wref.max())
        super().__init__(path, param_names, points, 'u.AA', 'u.erg/u.s/u.cm**2/u.AA',
                         wl_range, air=True, name='Koester_DA')
        self.wref, _ = np.loadtxt(self._filename(20000, 8.0), unpack=True)
        self.ind = (self.wref >= self.wl_range[0]) & (self.wref <= self.wl_range[1])
        self.wl = self.wref[self.ind]
        self.base = os.path.expandvars(self.path)
        self.rname = os.path.join(self.base, 'Koester_{0:0>5.0f}_{1:.2f}.fits')

    def _filename(self, teff, log_g):
        fn = f"da{int(teff):05d}_{int(100*log_g):03d}.dk.dat.txt"
        return os.path.join(self.path, fn)

    def _read_file(self, teff, log_g):
        fn = self._filename(teff, log_g)
        w, f = np.loadtxt(fn, unpack=True)

        func = InterpolatedUnivariateSpline(w, f, k=5)
        return func(self.wref)

    def load_flux(self, parameters, header=False, norm=True):
        '''
        Load a the flux and header information.

        :param parameters: stellar parameters
        :type parameters: dict

        :raises C.GridError: if the file cannot be found on disk.

        :returns: tuple (flux_array, header_dict)
        '''
        self.check_params(parameters)
        teff, log_g = parameters

        #Still need to check that file is in the grid, otherwise raise a C.GridError
        try:
            f = self._read_file(teff, log_g)
        except OSError:
            raise C.GridError(f"No file for Teff={teff} and log g={log_g}")
        if norm:
            f /= np.average(f)
        header = {}
        header['norm'] = norm
        header['air'] = self.air
        if header:
            return (f[self.ind], header)

        return f[self.ind]
