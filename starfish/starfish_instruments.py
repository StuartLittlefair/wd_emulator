from Starfish.grid_tools import instruments


class FORS2_1200B(instruments.Instrument):
    """
    FORS2 with the 1200B+97 grating and a 0.7" slit giving R=2000
    """
    def __init__(self):
        super().__init__('FORS2', FWHM=150, wl_range=(3660, 5110), oversampling=6)
