from blobmodel import Model, show_model, BlobShapeImpl, DefaultBlobFactory
import numpy as np

# velocities
vx = 0
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
    blob_alignment=False,
)

# blob tilting
theta = np.pi / 4
bf.set_theta_setter(
    lambda: theta
)  # If you want to stochasticity in the angle theta, you can do this thanks to the lambda

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
)
ds = bm.make_realization(speed_up=True, error=1e-2)
# show animation and save as gif
show_model(dataset=ds, interval=100, gif_name="2d_animation.gif", fps=10)
