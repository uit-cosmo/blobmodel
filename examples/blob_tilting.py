from blobmodel import Model, show_model, BlobShapeImpl, DefaultBlobFactory
import numpy as np

# velocities
vx = 1
vy = 1

# sizes
wx = 3
wy = 1

bf = DefaultBlobFactory(
    A_dist="deg",
    wx_dist="deg",
    vx_dist="deg",
    vy_dist="deg",
    vy_parameter=vy,
    vx_parameter=vx,
    wx_parameter=wx,
    wy_parameter=wy,
    blob_alignment=True,  # Blobs are aligned
)

# blob tilting
theta = np.pi / 4
# Using a lambda function to set the tilt angle theta, allows us to set a distribution for tilt angles if desired.
# In this case we use a degenerate distribution.
bf.set_theta_setter(lambda: theta)

# create data
bm = Model(
    Nx=100,
    Ny=100,
    Lx=10,
    Ly=10,
    dt=0.1,
    T=20,
    periodic_y=True,
    blob_shape=BlobShapeImpl("rectangle", "rectangle"),
    num_blobs=100,
    t_drain=1e10,
    blob_factory=bf,
    t_init=10,
)
ds = bm.make_realization(speed_up=True, error=1e-2)
# show animation and save as gif
show_model(
    dataset=ds,
    interval=100,
    gif_name="blob_alignment_true.gif",
    fps=10,
    initial_time=10,
)
