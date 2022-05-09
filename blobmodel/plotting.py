import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import xarray as xr
from matplotlib import animation


def show_model(
    dataset: xr.Dataset,
    variable: str = "n",
    interval: int = 100,
    save: bool = False,
    gif_name: str = "blobs.gif",
    fps: int = 10,
) -> None:
    """Show animation of Model output.

    Parameters
    ----------
    dataset: xarray Dataset,
        Model data
    variable: str, optional
        variable to be animated
    interval: int, optional
        time interval between frames in ms
    save: bool, optional
        if True save animation as gif
    gif_name: str, optional
        set name for gif
    fps: int, optional
        set fps for gif
    """
    fig = plt.figure()

    dt = dataset.t.values[1] - dataset.t.values[0]

    frames = []

    for timestep in dataset.t.values:
        frame = dataset[variable].sel(t=timestep).values
        frames.append(frame)

    def animate_1d(i: int) -> None:
        x = dataset.x
        y = frames[i]
        line.set_data(x, y)
        plt.title(f"t = {i*dt:.2f}")

    def animate_2d(i: int) -> None:
        arr = frames[i]
        vmax = np.max(arr)
        vmin = np.min(arr)
        im.set_data(arr)
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

    if save:
        ani.save(gif_name, writer="ffmpeg", fps=fps)
    plt.show()


def _setup_1d_plot(dataset, variable):
    ax = plt.axes(xlim=(0, dataset.x[-1]), ylim=(0, dataset[variable].max()))
    tx = ax.set_title(r"$t = 0$")
    (line,) = ax.plot([], [], lw=2)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(rf"${variable}$")
    return line, tx


def _setup_2d_plot(fig, cv0):
    ax = fig.add_subplot(111)
    tx = ax.set_title("t = 0")
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", "5%", "5%")
    im = ax.imshow(cv0, origin="lower")
    fig.colorbar(im, cax=cax)
    return im, tx
