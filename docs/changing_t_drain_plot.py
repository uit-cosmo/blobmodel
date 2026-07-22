from blobmodel import (
    BlobShapeEnum,
    BlobShapeImpl,
    DefaultBlobFactory,
    DistributionEnum,
    Geometry,
    Model,
)
import matplotlib.pyplot as plt
import numpy as np

t_drain = np.linspace(2, 1, 100)

tmp = Model(
    geometry=Geometry(Nx=100, Ny=1, Lx=10, Ly=0, dt=1, T=1000, periodic_y=False),
    blob_shape=BlobShapeImpl(BlobShapeEnum.exp),
    num_blobs=10000,
    blob_factory=DefaultBlobFactory(
        A_dist=DistributionEnum.deg,
        vy_dist=DistributionEnum.zeros,
        t_drain=t_drain,
    ),
)

ds_changing_t_drain = tmp.make_realization(speed_up=False)

tmp = Model(
    geometry=Geometry(Nx=100, Ny=1, Lx=10, Ly=0, dt=1, T=1000, periodic_y=False),
    blob_shape=BlobShapeImpl(BlobShapeEnum.exp),
    num_blobs=10000,
    blob_factory=DefaultBlobFactory(
        A_dist=DistributionEnum.deg,
        vy_dist=DistributionEnum.zeros,
        t_drain=2,
    ),
)

ds_constant_drain = tmp.make_realization(speed_up=False)


def plot_cahnging_t_drain(changing, constant):
    changing.n.isel(y=0).mean(dim=("t")).plot(label="decreasing t_drain")
    constant.n.isel(y=0).mean(dim=("t")).plot(label="constant t_drain")
    plt.yscale("log")
    plt.legend()
    plt.show()


plot_cahnging_t_drain(ds_changing_t_drain, ds_constant_drain)
