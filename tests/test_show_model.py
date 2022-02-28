from unittest.mock import patch
from blobmodel import Model, show_model
import warnings


bm = Model(
    Nx=10,
    Ny=10,
    Lx=10,
    Ly=10,
    dt=0.1,
    T=1,
    periodic_y=False,
    blob_shape="exp",
    num_blobs=1,
)

# create data
ds = bm.make_realization()

# warnings are supressed since plt complains about animation blocked
@patch("matplotlib.pyplot.show")
def test_plot_fn(mock_show):
    warnings.filterwarnings("ignore")
    show_model(dataset=ds, interval=100, save=False, fps=10)


test_plot_fn()
