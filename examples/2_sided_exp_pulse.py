from blobmodel import (
    Model,
    show_model,
    DefaultBlobFactory,
    BlobShapeImpl,
    DistributionEnum,
    BlobShapeEnum,
)
import matplotlib.pyplot as plt


bf = DefaultBlobFactory(
    A_dist=DistributionEnum.deg,
    wx_dist=DistributionEnum.deg,
    spx_dist=DistributionEnum.deg,
    spy_dist=DistributionEnum.deg,
    shape_param_x_parameter=0.5,
    shape_param_y_parameter=0.5,
)

bm = Model(
    Nx=100,
    Ny=100,
    Lx=10,
    Ly=10,
    dt=0.1,
    T=10,
    num_blobs=10,
    blob_shape=BlobShapeImpl(BlobShapeEnum.double_exp, BlobShapeEnum.double_exp),
    t_drain=1e10,
    blob_factory=bf,
)

ds = bm.make_realization(speed_up=True, error=1e-10)

show_model(ds)

plt.show()
