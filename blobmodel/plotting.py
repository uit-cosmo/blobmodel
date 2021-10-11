import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from tqdm import tqdm
from blobmodel import Model


def show_model(
    model: Model,
    interval: int = 100,
    save: bool = False,
    gif_name: str = "2d_blobs.gif",
    fps: int = 10,
) -> None:
    """
        show animation of Model

        Parameters
        ----------
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
    ax = fig.add_subplot(111)
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", "5%", "5%")

    frames = []

    __xx, __yy = np.meshgrid(model.x, model.y)

    for t in tqdm(model.t, desc="Creating frames for animation"):
        curVals = np.zeros(shape=(model.Ny, model.Nx))
        for b in model.get_blobs():
            curVals += b.discretize_blob(
                x=__xx, y=__yy, t=t, periodic_y=model.periodic_y, Ly=model.Ly
            )
        frames.append(curVals)

    cv0 = frames[0]
    im = ax.imshow(cv0, origin="lower")
    fig.colorbar(im, cax=cax)
    tx = ax.set_title("t = 0")

    def animate(i: int) -> None:
        arr = frames[i]
        vmax = np.max(arr)
        vmin = np.min(arr)
        im.set_data(arr)
        im.set_clim(vmin, vmax)
        tx.set_text(f"t = {i*model.dt:.2f}")

    ani = FuncAnimation(fig, animate, frames=model.t.size, interval=interval)
    if save:
        ani.save(gif_name, writer="ffmpeg", fps=fps)
    plt.show()
