from blobmodel import Model, show_model, DefaultBlobFactory, BlobShapeImpl
import matplotlib.pyplot as plt
import numpy as np


bf = DefaultBlobFactory(
    A_dist="deg",
    vy_dist="deg",
    wx_dist="deg",
    spx_dist="deg",
    spy_dist="deg",
    shape_param_x_parameter=0.5,
    shape_param_y_parameter=0.5,
    blob_alignment=False,
)

bf.set_theta_setter(lambda: np.random.uniform(-np.pi / 2, np.pi / 2))
# bf.set_theta_setter(lambda: np.pi/4)

bm = Model(
    Nx=100,
    Ny=100,
    Lx=10,
    Ly=10,
    dt=0.1,
    T=10,
    num_blobs=10,
    blob_shape=BlobShapeImpl("exp", "2-exp"),
    periodic_y=True,
    t_drain=1e10,
    blob_factory=bf,
)

ds = bm.make_realization(speed_up=True, error=1e-10)

show_model(ds)

plt.show()
