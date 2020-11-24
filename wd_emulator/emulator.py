import itertools

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from sklearn.decomposition import PCA
from .visualisation import imshow


class Emulator:

    def __init__(self, wavelengths, teffs, gravities, spectra):
        """
        Spectral Emulator using PCA.

        The model grid is decomposed into a series of eigenspectra, such
        that each spectrum on the grid is a weighted sum of eigenspectra.

        The weights of each component will vary smoothly with log g and Teff
        so by interpolating the weights to recreate spectra at positions
        not covered by the grid, we are interpolating in a smoother space
        than interpolating the spectra themselves. This leads to lower error.

        You could interpolate the weights any way you choose. Here we use a
        Delaunay triangulation (`~scipy.interpolate.LinearNDInterpolator`).

        Parameters
        ----------
        wavelengths: `np.ndarray`
            wavelengths on which the model grid is calculated
        teffs: `np.ndarray`
            the effective temperatures at which the grid is sampled
        gravities: `np.ndarray`
            the log g values at which the grid is sampled
        spectra: `np.ndarray`
            the model spectra. Should be of shape (nteffs, ngrav, nspec)
        """
        if (teffs.size, gravities.size, wavelengths.size) != spectra.shape:
            raise ValueError('shapes of inputs do not match')

        nteffs, nlogg, nwavs = spectra.shape
        self.model_fluxes = spectra.reshape(nteffs*nlogg, nwavs)
        self.wavelengths = wavelengths

        # a nspectra by 2 array of logg and teff
        log_teff = np.log10(teffs)
        self.x = np.array(list(itertools.product(log_teff, gravities)))
        self.nteffs = nteffs
        self.nlogg = nlogg
        # Normalize to an average of 1 to remove overall brightness changes
        self._normalised_fluxes = (self.model_fluxes /
                                   self.model_fluxes.mean(1, keepdims=True))
        self._pca = None
        self._gps = None
        self._interpolator = None
        self._weight_interpolator = None

    def linear_interpolation(self, params):
        """
        Linear barycentric interpolation of the model grid.

        This is a simple alternative to PCA based emulation, using
        `~scipy.interpolate.LinearNDInterpolator`. The grid is triangulated
        using Qhull (http://www.qhul.org/) and then the resulting spectrum
        is found via linear barycentric interpolation on each triangle.

        This is much more accurate than the commonly used bi-linear interpolation.

        Parameters
        ----------
        params: ~np.ndarray
            A pair of log_10(Teff) and log_g values. Can also be a large (N, 2)
            array if you want to calculate spectra at many points simultaneously.

        Returns
        -------
        spectrum: ~np.ndarray
            The interpolated spectrum or spectra.
        """
        if self._interpolator is None:
            self._interpolator = LinearNDInterpolator(
                self.x,
                self._normalised_fluxes,
                rescale=True)
        return self._interpolator(params).squeeze()

    def run_pca(self, target_variance=0.995, **pca_kwargs):
        """
        Perform PCA on model grid.

        We allow the PCA to choose the number of components to
        explain a target fraction o fthe total variance within
        the model grid.

        Parameters
        ----------
        target_variance: float
            variance to aim for
        pca_kwargs: dict
            any additional arguments to pass directly to sklearn's PCA
            class
        """
        default_pca_kwargs = dict(n_components=target_variance,
                                  svd_solver="full", whiten=True)
        default_pca_kwargs.update(pca_kwargs)
        self._pca = PCA(**default_pca_kwargs)
        self._pca_weights = self._pca.fit_transform(self._normalised_fluxes)
        self.eigenspectra = self._pca.components_
        self.ncomps = self._pca.n_components_
        exp_var = self._pca.explained_variance_ratio_.sum()

        # save this quantity for later use in reconstructing spectra
        self._X = (np.sqrt(self._pca.explained_variance_[:, np.newaxis]) *
                   self.eigenspectra)
        return exp_var

    def train(self):
        """
        Train GPs to interpolate weight array.

        Currently struggling to train properly
        """
        raise NotImplementedError('cant train GPs yet')
        if self._pca is None:
            raise RuntimeError('must run pca first')

    def get_index(self, params):
        """
        Given a list of stellar parameters (corresponding to a grid point),
        deliver the index that corresponds to the
        entry in the fluxes, grid_points, and weights.

        Parameters
        ----------
        params : array_like
            The stellar parameters

        Returns
        -------
        index : int
        """
        params = np.atleast_2d(params)
        marks = np.abs(self.x - np.expand_dims(params, 1)).sum(axis=-1)
        return marks.argmin(axis=1).squeeze()

    def _predict_weights(self, pars):
        """
        Linear interpolation of weight maps.

        Uses `~scipy.interpolate.LinearNDInterpolator`. The grid is triangulated
        using Qhull (http://www.qhul.org/) and then the resulting spectrum
        is found via linear barycentric interpolation on each triangle.

        This is much more accurate than the commonly used bi-linear interpolation.
        """

        if self._weight_interpolator is None:
            self._weight_interpolator = LinearNDInterpolator(
                self.x, self._pca_weights, rescale=True
            )
        return self._weight_interpolator(pars)

    def __call__(self, pars):
        """
        Emulate the spectrum by interpolating weight maps to find eigenvector weights
        at this temperature and gravity.

        Parameters
        ----------
        params: ~np.ndarray
            A pair of log_10(Teff) and log_g values. Can also be a large (N, 2)
            array if you want to calculate spectra at many points simultaneously.

        Returns
        -------
        spectrum: ~np.ndarray
            The interpolated spectrum or spectra.
        """

        # reshape pars into (N, 2) grid of log g, teff. For now assume correct
        # shape
        weights = self._predict_weights(pars)
        spectra = weights @ self._X + self._pca.mean_
        return spectra.squeeze()

    def plot_training_results(self, logg_idx):
        """
        Plot the training results at a slice of log g.

        Plots the actual weights found via PCA and the predicted weights
        from interpolation on a finer grid.
        """
        if self.ncomps < 4:
            fig, axes = plt.subplots(
                self.ncomps, 1, sharex=True, figsize=(8, (self.ncomps - 1) * 2)
            )
        else:
            fig, axes = plt.subplots(
                int(np.ceil(self.ncomps / 2)), 2,
                sharex=True,
                figsize=(13, (self.ncomps - 1) * 2)
            )
        axes = np.ravel(np.array(axes).T)
        [ax.set_ylabel(f"$w_{i}$") for i, ax in enumerate(axes)]

        logT = np.unique(self.x[:, 0])
        logg = np.unique(self.x[:, 1])
        params = np.array(list(itertools.product(logT, logg[logg_idx:logg_idx+1])))
        idxs = np.array([self.get_index(p) for p in params])
        weights = self._pca_weights[idxs.astype("int")].T
        for i, w in enumerate(weights):
            axes[i].plot(logT, w, ".")

        Ttest = np.linspace(logT.min(), logT.max(), 300)
        Xtest = np.array(list(itertools.product(Ttest, logg[logg_idx:logg_idx+1])))
        mus = self._predict_weights(Xtest)
        for i, m in enumerate(mus.T):
            axes[i].plot(Ttest, m)
        axes[-1].set_xlabel("$\log_{10}$ T (K)")
        plt.suptitle(
            f"Weights for fixed $\log g={logg[logg_idx]:.2f}$", fontsize=20
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    def plot_component_weights(self):
        """
        Plot the component weight maps.
        """
        if self.ncomps < 4:
            fig, axes = plt.subplots(
                self.ncomps, 1, sharex=True, figsize=(8, (self.ncomps - 1) * 2)
            )
        else:
            fig, axes = plt.subplots(
                int(np.ceil(self.ncomps / 2)), 2,
                sharex=True,
                figsize=(13, (self.ncomps - 1) * 2)
            )
        axes = np.ravel(np.array(axes).T)
        [ax.set_title(f"$w_{i}$") for i, ax in enumerate(axes)]
        min_t, min_g = self.x.min(axis=0)
        max_t, max_g = self.x.max(axis=0)
        for i, ax in enumerate(axes):
            image = self._pca_weights[..., i].reshape(self.nteffs, self.nlogg)
            imshow(image,
                   axis=ax, aspect='auto',
                   extent=(min_g, max_g, min_t, max_t), cmap='viridis')
            ax.set_xlabel('log g')
            ax.set_ylabel('$\log_{10}$ Teff (K)')
        fig.tight_layout()
