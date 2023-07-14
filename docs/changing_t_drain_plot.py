from blobmodel import Model, DefaultBlobFactory
import matplotlib.pyplot as plt
import numpy as np

bf = DefaultBlobFactory(A_dist="deg", wx_dist="deg", vx_dist="deg", vy_dist="zeros")

t_drain = np.linspace(2, 1, 100)

tmp = Model(
    Nx=100,
    Ny=1,
    Lx=10,
    Ly=0,
    dt=1,
    T=1000,
    blob_shape="exp",
    t_drain=t_drain,
    periodic_y=False,
    num_blobs=10000,
    blob_factory=bf,
)

ds_changing_t_drain = tmp.make_realization(speed_up=False)

tmp = Model(
    Nx=100,
    Ny=1,
    Lx=10,
    Ly=0,
    dt=1,
    T=1000,
    blob_shape="exp",
    t_drain=2,
    periodic_y=False,
    num_blobs=10000,
    blob_factory=bf,
)

ds_constant_drain = tmp.make_realization(speed_up=False)


def plot_cahnging_t_drain(changing, constant):
    changing.n.isel(y=0).mean(dim=("t")).plot(label="decreasing t_drain")
    constant.n.isel(y=0).mean(dim=("t")).plot(label="constant t_drain")
    plt.yscale("log")
    plt.legend()
    plt.show()


plot_cahnging_t_drain(ds_changing_t_drain, ds_constant_drain)
