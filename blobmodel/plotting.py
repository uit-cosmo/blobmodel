"""This module provides functions to create and display animations of model output."""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import xarray as xr
from matplotlib import animation
from typing import Union, Any


def show_model(
    dataset: xr.Dataset,
    variable: str = "n",
    interval: int = 100,
    gif_name: Union[str, None] = None,
    fps: int = 10,
) -> None:
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
        If not None, save the animation as a GIF and name it acoridingly.
    fps : int, optional
        Set the frames per second for the saved GIF (default: 10).

    Returns
    -------
    None

    Notes
    -----
    - This function chooses between a 1D and 2D visualizations based on the dimensionality of the dataset.

    """
    fig = plt.figure()

    dt = dataset.t.values[1] - dataset.t.values[0]

    frames = []

    for timestep in dataset.t.values:
        frame = dataset[variable].sel(t=timestep).values
        frames.append(frame)

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
        x = dataset.x
        y = frames[i]
        line.set_data(x, y)
        plt.title(f"t = {i*dt:.2f}")

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
        arr = frames[i]
        vmax = np.max(arr)
        vmin = np.min(arr)
        im.set_data(arr)
        im.set_extent((dataset.x[0], dataset.x[-1], dataset.y[0], dataset.y[-1]))
        im.set_clim(vmin, vmax)
        tx.set_text(f"t = {i*dt:.2f}")

    if dataset.y.size == 1:
        line, tx = _setup_1d_plot(dataset=dataset, variable=variable)
        ani = animation.FuncAnimation(
            fig, animate_1d, frames=dataset["t"].values.size, interval=interval
        )
    else:
        im, tx = _setup_2d_plot(fig=fig, cv0=frames[0])
        ani = animation.FuncAnimation(
            fig, animate_2d, frames=dataset["t"].values.size, interval=interval
        )

    if gif_name:
        ani.save(gif_name, writer="ffmpeg", fps=fps)
    plt.show()


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
    ax = plt.axes(xlim=(0, dataset.x[-1]), ylim=(0, dataset[variable].max()))
    tx = ax.set_title(r"$t = 0$")
    (line,) = ax.plot([], [], lw=2)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(rf"${variable}$")
    return line, tx


def _setup_2d_plot(fig, cv0):
    """
    Set up a 2D plot for the animation.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object for the plot.
    cv0 : numpy.ndarray
        Initial 2D array for the plot.

    Returns
    -------
    im : matplotlib.image.AxesImage
        Image object representing the plot.
    tx : matplotlib.text.Text
        Text object for the plot title.

    """
    ax = fig.add_subplot(111)
    tx = ax.set_title("t = 0")
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", "5%", "5%")
    im = ax.imshow(cv0, origin="lower")
    fig.colorbar(im, cax=cax)
    return im, tx
