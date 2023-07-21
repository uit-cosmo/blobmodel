from blobmodel import Model, show_model, DefaultBlobFactory, BlobShapeImpl
import matplotlib.pyplot as plt
import numpy as np


bf = DefaultBlobFactory(
    A_dist="deg",
    wx_dist="deg",
    spx_dist="deg",
    spy_dist="deg",
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
    blob_shape=BlobShapeImpl("2-exp", "2-exp"),
    periodic_y=True,
    t_drain=1e10,
    blob_factory=bf,
)

ds = bm.make_realization(speed_up=True, error=1e-10)

show_model(ds)

plt.show()
