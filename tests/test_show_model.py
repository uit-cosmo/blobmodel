import subprocess
import sys
from unittest.mock import patch

import matplotlib.pyplot as plt
import pytest
from blobmodel import Geometry, Model, show_model
import warnings

bm_2d = Model(
    geometry=Geometry(Nx=10, Ny=10, Lx=10, Ly=10, dt=0.1, T=1, periodic_y=False),
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
    geometry=Geometry(Nx=10, Ny=1, Lx=10, Ly=1, dt=0.1, T=1, periodic_y=False),
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
def test_plot_1d_squeezed(mock_show):
    """
    Checks that show model runs on true 1D output (Ly = 0), which has no y
    dimension at all.
    """
    warnings.filterwarnings("ignore")
    bm = Model(
        geometry=Geometry(Nx=10, Ny=1, Lx=10, Ly=0, dt=0.1, T=1, periodic_y=False),
        num_blobs=1,
        one_dimensional=True,
    )
    show_model(dataset=bm.make_realization(), interval=100, gif_name=None, fps=10)
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


@patch("matplotlib.pyplot.show")
def test_1d_plot_honors_domain_origin(mock_show):
    """
    The 1D plot x-limits must start at the domain origin (x0 may be nonzero),
    not at a hardcoded 0.
    """
    warnings.filterwarnings("ignore")
    bm = Model(
        geometry=Geometry(Nx=10, Ny=1, Lx=10, Ly=0, dt=0.1, T=1, x0=100.0),
        num_blobs=1,
        one_dimensional=True,
    )
    ds = bm.make_realization()
    show_model(dataset=ds, show=False)
    xlim = plt.gca().get_xlim()
    assert xlim[0] == pytest.approx(float(ds.x[0]))
    assert xlim[1] == pytest.approx(float(ds.x[-1]))
    plt.close("all")


def test_imaging_layout_raises_clear_error():
    """
    Imaging-layout datasets (frames/time, no t coordinate) are not plottable
    with show_model; the failure must be a clear ValueError instead of an
    AttributeError from deep inside.
    """
    ds_imaging = bm_2d.make_realization(layout="imaging")
    with pytest.raises(ValueError, match="imaging"):
        show_model(dataset=ds_imaging, variable="frames")


def test_import_does_not_pull_matplotlib():
    """
    Importing blobmodel must not import matplotlib.pyplot; headless
    environments that only simulate should not pay for a display backend.
    """
    code = "import sys; import blobmodel; assert 'matplotlib.pyplot' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
