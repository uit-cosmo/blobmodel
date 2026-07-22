from blobmodel import (
    Geometry,
    Model,
    show_model,
    DefaultBlobFactory,
    BlobShapeImpl,
    BlobShapeEnum,
    DistributionEnum,
)

# Example of a one-dimensional realization and animation.

bf = DefaultBlobFactory(vy_dist=DistributionEnum.zeros, t_drain=10)

bm = Model(
    geometry=Geometry(Nx=100, Ny=1, Lx=10, Ly=0, dt=0.1, T=10, periodic_y=False),
    blob_shape=BlobShapeImpl(BlobShapeEnum.exp, BlobShapeEnum.exp),
    num_blobs=20,
    blob_factory=bf,
    one_dimensional=True,
)

ds = bm.make_realization(speed_up=True, error=1e-2)
show_model(dataset=ds, interval=100, gif_name="1d_animation.gif")
