from blobmodel import Model, show_model, DefaultBlobFactory
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

# here you can define your custom parameter distributions
bf = DefaultBlobFactory(A_dist="deg", W_dist="deg", vx_dist="deg", vy_dist="zeros")

bm = Model(
    Nx=100,
    Ny=1,
    Lx=10,
    Ly=0,
    dt=0.1,
    T=20,
    periodic_y=False,
    blob_shape="exp",
    num_blobs=100,
    t_drain=10,
    blob_factory=bf,
)

ds = bm.make_realization(speed_up=True, error=1e-2)
print(ds)


# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, ds.x[-1]), ylim=(0, ds.n.max()))
(line,) = ax.plot([], [], lw=2)

frames = []

for timestep in ds.t.values:
    frame = ds["n"].sel(t=timestep).values
    frames.append(frame)

tx = ax.set_title("t = 0")

dt = ds.t.values[1] - ds.t.values[0]
# animation function.  This is called sequentially
def animate(i: int) -> None:
    x = ds.x
    y = frames[i]
    line.set_data(x, y)
    plt.title(f"t = {i*dt:.2f}")


plt.xlabel("x")
plt.ylabel("n")
# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(
    fig,
    animate,
    frames=ds["t"].values.size,
    interval=100,
)
plt.show()
