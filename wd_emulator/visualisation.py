from matplotlib import pyplot as plt
from astropy.visualization import (AsymmetricPercentileInterval, ImageNormalize,
                                   SqrtStretch, LogStretch)


def imshow(data, axis=None, plo=2, phi=98, stretch=None, **kwargs):

    if stretch is None:
        norm = ImageNormalize(
            data,
            interval=AsymmetricPercentileInterval(plo, phi)
        )
    elif stretch == 'sqrt':
        norm = ImageNormalize(
            data,
            interval=AsymmetricPercentileInterval(plo, phi),
            stretch=SqrtStretch()
        )
    elif stretch == 'log':
        norm = ImageNormalize(
            data,
            interval=AsymmetricPercentileInterval(plo, phi),
            stretch=LogStretch()
        )
    else:
        raise ValueError(f'stretch of {stretch} is not supported')
    if axis is None:
        return plt.imshow(data, origin='lower', norm=norm, **kwargs)
    else:
        return axis.imshow(data, origin='lower', norm=norm, **kwargs)
