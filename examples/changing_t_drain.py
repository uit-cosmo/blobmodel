from blobmodel import (
    Geometry,
    Model,
    DefaultBlobFactory,
    DistributionEnum,
    BlobShapeEnum,
    BlobShapeImpl,
)
import matplotlib.pyplot as plt
import numpy as np

# Example of use of a spatially varying t_drain parameter. The results are compared with the case of constant
# t_drain.

t_drain = np.linspace(2, 1, 100)

decreasing_t_drain = Model(
    geometry=Geometry(Nx=100, Ny=1, Lx=10, Ly=0, dt=1, T=1000, periodic_y=False),
    blob_shape=BlobShapeImpl(BlobShapeEnum.exp, BlobShapeEnum.gaussian),
    num_blobs=10000,
    blob_factory=DefaultBlobFactory(t_drain=t_drain)
    .set_sampler("amplitude", DistributionEnum.deg)
    .set_sampler("vy", DistributionEnum.zeros),
)

ds_decreasing = decreasing_t_drain.make_realization(truncation_error=1e-2)
ds_decreasing.n.mean(dim=("t")).plot(label="decreasing t_drain")

constant_t_drain = Model(
    geometry=Geometry(Nx=100, Ny=1, Lx=10, Ly=0, dt=1, T=1000, periodic_y=False),
    blob_shape=BlobShapeImpl(BlobShapeEnum.exp, BlobShapeEnum.gaussian),
    num_blobs=10000,
    blob_factory=DefaultBlobFactory(t_drain=2)
    .set_sampler("amplitude", DistributionEnum.deg)
    .set_sampler("vy", DistributionEnum.zeros),
)

ds_constant = constant_t_drain.make_realization(truncation_error=1e-2)
ds_constant.n.mean(dim=("t")).plot(label="constant t_drain")

plt.yscale("log")
plt.legend()
plt.show()
