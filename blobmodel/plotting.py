"""This module provides functions to create and display animations of model output."""

from typing import Union, Any, TYPE_CHECKING
import xarray as xr

if TYPE_CHECKING:
    from matplotlib import animation


def show_model(
    dataset: xr.Dataset,
    variable: str = "n",
    interval: int = 100,
    gif_name: Union[str, None] = None,
    fps: int = 10,
    show: bool = True,
) -> "animation.FuncAnimation":
    """
    Creates an animation that shows the evolution of a specific variable over time.

    Parameters
    ----------
    dataset : xr.Dataset
        Model data.
    variable : str, optional
        Variable to be animated (default: "n").
    interval : int, optional
        Time interval between frames in milliseconds (default: 100).
    gif_name : str, optional
        If not None, save the animation as a GIF and name it accordingly.
    fps : int, optional
        Set the frames per second for the saved GIF (default: 10).
    show : bool, optional
        If True (the default), display the animation in an interactive window
        with ``plt.show()``, which blocks until the window is closed. Set to
        False in scripts and CI, e.g. when only saving a GIF.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation object. Keep a reference to it as long as the animation
        should stay alive.

    Notes
    -----
    - This function chooses between a 1D and 2D visualizations based on the dimensionality of the dataset.
    - The color scale is fixed to the global minimum and maximum of `variable`
      over the whole dataset.

    """
    import matplotlib.pyplot as plt
    from matplotlib import animation

    fig = plt.figure()

    data = dataset[variable]
    t_values = dataset.t.values

    def animate_1d(i: int) -> Any:
        """
        Create the 1D plot for each frame of the animation.

        Parameters
        ----------
        i : int
            Frame index.

        Returns
        -------
        None

        """
        line.set_data(dataset.x.values, data.isel(t=i).values.ravel())
        tx.set_text(f"t = {t_values[i]:.2f}")

    def animate_2d(i: int) -> Any:
        """
        Create the 2D plot for each frame of the animation.

        Parameters
        ----------
        i : int
            Frame index.

        Returns
        -------
        None

        """
        im.set_data(data.isel(t=i).values)
        tx.set_text(f"t = {t_values[i]:.2f}")

    # 1D output (Ly = 0) has no y dimension at all; a 2D dataset with a
    # single y point is also shown as a line plot.
    if "y" not in dataset.dims or dataset.y.size == 1:
        line, tx = _setup_1d_plot(dataset=dataset, variable=variable)
        ani = animation.FuncAnimation(
            fig, animate_1d, frames=t_values.size, interval=interval
        )
    else:
        im, tx = _setup_2d_plot(fig=fig, dataset=dataset, variable=variable)
        ani = animation.FuncAnimation(
            fig, animate_2d, frames=t_values.size, interval=interval
        )

    if gif_name:
        ani.save(gif_name, writer="ffmpeg", fps=fps)
    if show:
        plt.show()
    return ani


def _setup_1d_plot(dataset, variable):
    """
    Set up a 1D plot for the animation.

    Parameters
    ----------
    dataset : xr.Dataset
        Model data.
    variable : str
        Variable to be animated.

    Returns
    -------
    line : matplotlib.lines.Line2D
        Line object representing the plot.
    tx : matplotlib.text.Text
        Text object for the plot title.

    """
    import matplotlib.pyplot as plt

    ax = plt.axes(xlim=(0, dataset.x[-1]), ylim=(0, dataset[variable].max()))
    tx = ax.set_title(r"$t = 0$")
    (line,) = ax.plot([], [], lw=2)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(rf"${variable}$")
    return line, tx


def _setup_2d_plot(fig, dataset, variable):
    """
    Set up a 2D plot for the animation.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object for the plot.
    dataset : xr.Dataset
        Model data.
    variable : str
        Variable to be animated.

    Returns
    -------
    im : matplotlib.image.AxesImage
        Image object representing the plot.
    tx : matplotlib.text.Text
        Text object for the plot title.

    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    data = dataset[variable]
    ax = fig.add_subplot(111)
    tx = ax.set_title("t = 0")
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", "5%", "5%")
    im = ax.imshow(
        data.isel(t=0).values,
        origin="lower",
        extent=(
            float(dataset.x[0]),
            float(dataset.x[-1]),
            float(dataset.y[0]),
            float(dataset.y[-1]),
        ),
        vmin=float(data.min()),
        vmax=float(data.max()),
    )
    fig.colorbar(im, cax=cax)
    return im, tx
