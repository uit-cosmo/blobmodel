import subprocess
import sys
from unittest.mock import patch

import matplotlib.pyplot as plt
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
    Checks that show model runs on a two-dimensional model and shows the plot
    by default.
    """
    warnings.filterwarnings("ignore")
    show_model(dataset=ds_2d, interval=100, gif_name=None, fps=10)
    assert mock_show.called
    plt.close("all")


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
    plt.close("all")


@patch("matplotlib.pyplot.show")
def test_show_false_does_not_block(mock_show):
    """
    show=False must not call plt.show(), so scripts and CI are not blocked.
    """
    ani = show_model(dataset=ds_2d, show=False)
    plt.gcf().canvas.draw()  # render a frame so the animation is not "unused"
    assert not mock_show.called
    plt.close("all")


def test_color_scale_is_global():
    """
    The 2D animation color limits are fixed to the global min/max of the
    variable, so the colorbar does not flicker between frames.
    """
    ani = show_model(dataset=ds_2d, show=False)
    fig = plt.gcf()
    fig.canvas.draw()  # render a frame so the animation is not "unused"
    im = fig.axes[0].images[0]
    assert im.get_clim() == (float(ds_2d.n.min()), float(ds_2d.n.max()))
    plt.close("all")


def test_import_does_not_pull_matplotlib():
    """
    Importing blobmodel must not import matplotlib.pyplot; headless
    environments that only simulate should not pay for a display backend.
    """
    code = "import sys; import blobmodel; assert 'matplotlib.pyplot' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
