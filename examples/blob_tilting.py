from blobmodel import (
    Model,
    show_model,
    BlobShapeImpl,
    DefaultBlobFactory,
    BlobShapeEnum,
    DistributionEnum,
)
import numpy as np

# Example with tilted blobs, to have something to compare we will first make a simulation with
# aligned blobs moving with velocities

vx = 1
vy = 1

# and sizes that make it easy to say whether or not they are tilted

wx = 3
wy = 1

bf = DefaultBlobFactory(
    A_dist=DistributionEnum.deg,
    vy_parameter=vy,
    vx_parameter=vx,
    wx_parameter=wx,
    wy_parameter=wy,
    blob_alignment=True,  # Blobs will be aligned with their velocities
)


# create data
bm = Model(
    Nx=100,
    Ny=100,
    Lx=10,
    Ly=10,
    dt=0.1,
    T=20,
    periodic_y=True,
    blob_shape=BlobShapeImpl(BlobShapeEnum.rect, BlobShapeEnum.rect),
    num_blobs=100,
    t_drain=1e10,
    blob_factory=bf,
    t_init=10,
)
ds = bm.make_realization(speed_up=True, error=1e-2)
# show animation and save as gif
show_model(dataset=ds, interval=100, gif_name="alignment_true.gif", fps=10)

# Now we do tilted blobs


bf = DefaultBlobFactory(
    A_dist=DistributionEnum.deg,
    vy_parameter=vy,
    vx_parameter=vx,
    wx_parameter=wx,
    wy_parameter=wy,
    blob_alignment=False,
)
# Blobs will NOT be aligned with their velocities, instead they will be tilted by an angle given by theta.

# blob tilting
theta = np.pi / 2
# Setting the angle with a lambda allows us to set a distribution of tilt angles. In this case we use a degenerate distribution:
bf.set_theta_setter(lambda: theta)

# create data, same as before
bm = Model(
    Nx=100,
    Ny=100,
    Lx=10,
    Ly=10,
    dt=0.1,
    T=20,
    periodic_y=True,
    blob_shape=BlobShapeImpl(BlobShapeEnum.rect, BlobShapeEnum.rect),
    num_blobs=100,
    t_drain=1e10,
    blob_factory=bf,
    t_init=10,
)
ds = bm.make_realization(speed_up=True, error=1e-2)
# show animation and save as gif
show_model(dataset=ds, interval=100, gif_name="alignment_false.gif", fps=10)
