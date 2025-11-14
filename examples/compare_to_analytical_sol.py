from blobmodel import (
    Model,
    DefaultBlobFactory,
    BlobShapeEnum,
    DistributionEnum,
    BlobShapeImpl,
)
import matplotlib.pyplot as plt
import numpy as np

# The time average of a one-dimensional model realization with constant horizontal velocities is compared
# with analytically derived results. See O. E. Garcia, et al.; Phys. Plasmas 1 May 2016; 23 (5): 052308. https://doi.org/10.1063/1.4951016

# use DefaultBlobFactory to define distribution functions fo random variables
bf = DefaultBlobFactory(
    A_dist=DistributionEnum.deg, vy_dist=DistributionEnum.zeros, vy_parameter=10
)
t_drain = 1e10

tmp = Model(
    Nx=10,
    Ny=10,
    Lx=10,
    Ly=10,
    dt=0.1,
    T=10000,
    blob_shape=BlobShapeImpl(BlobShapeEnum.gaussian, BlobShapeEnum.gaussian),
    t_drain=1e10,
    periodic_y=True,
    num_blobs=10000,
    blob_factory=bf,
    one_dimensional=False,
    t_init=10,
)

ds = tmp.make_realization(speed_up=True, error=1e-10)

mean = ds.n.mean(dim="t").values

fig, ax = plt.subplots()

im = ax.imshow(mean, clim=(0, np.max(mean)))
plt.colorbar(im, ax=ax)

plt.show()
