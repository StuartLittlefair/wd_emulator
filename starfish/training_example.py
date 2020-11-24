from plotting import plot
from starfish_instruments import FORS2_1200B
from starfish_tools import KoesterGridInterface
from Starfish.grid_tools import HDF5Creator
from Starfish.emulator import Emulator


# make a grid interface to the raw Koester spectra
# spectra are normalised
grid = KoesterGridInterface('.')

# make a HFD5 file with the model grid interpolated onto instrumental resolution
# if we are interested in narrow range of parameters, we can make the grid smaller here
ranges = [[6000, 30000], [7.0, 9.0]]

if ranges is not None:
    creator = HDF5Creator(grid, "Koester_FORS2_grid.hdf5",
                          instrument=FORS2_1200B(),
                          ranges=ranges)
else:
    creator = HDF5Creator(grid, "Koester_FORS2_grid.hdf5", instrument=FORS2_1200B())

# this uses the grid to load in and resample all the spectra, saving to hdf5 file
creator.process_grid()

# create a spectral emulator. This will do the PCA on the model grid
# you can specify the number of components. A number <1 will choose
# the number of PCA components to explain the variance, so
# n_components=0.99 will use sufficient eigenspectra to explain 99% of variance
emu = Emulator.from_grid("Koester_FORS2_grid.hdf5", n_components=0.995)

# now we have to train the emulator, which can take a while.
# This is fitting Gaussian Processes to the weights of each eigenspectrum
# as a function of log g and Teff. By using GPs rather than polynomials we
# can get a decent estimate of the the interpolation error

# training via the default Nelder-Mead algorithm is a bit finicky, so use
# Powell, which will get use close, and then Nelfer-Mead to optimise
emu.train(method='Powell')
emu.train()

plot(emu)
