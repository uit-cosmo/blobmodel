from unittest.mock import patch
from blobmodel import Model, show_model
import warnings


bm_2d = Model(
    Nx=10,
    Ny=10,
    Lx=10,
    Ly=10,
    dt=0.1,
    T=1,
    periodic_y=False,
    num_blobs=1,
)

# create data
ds_2d = bm_2d.make_realization()


# warnings are suppressed since plt complains about animation blocked
@patch("matplotlib.pyplot.show")
def test_plot_2d(mock_show):
    """
    Checks that show model runs on a two-dimensional model.
    """
    warnings.filterwarnings("ignore")
    show_model(dataset=ds_2d, interval=100, gif_name=None, fps=10)


bm_1d = Model(
    Nx=10,
    Ny=1,
    Lx=10,
    Ly=1,
    dt=0.1,
    T=1,
    periodic_y=False,
    num_blobs=1,
)

# create data
ds_1d = bm_1d.make_realization()


# warnings are supressed since plt complains about animation blocked
@patch("matplotlib.pyplot.show")
def test_plot_1d(mock_show):
    """
    Checks that show model runs on a one-dimensional model.
    """
    warnings.filterwarnings("ignore")
    show_model(dataset=ds_1d, interval=100, gif_name=None, fps=10)
