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
bf = DefaultBlobFactory(A_dist=DistributionEnum.deg, vy_dist=DistributionEnum.zeros)

tmp = Model(
    Nx=20,
    Ny=1,
    Lx=10,
    Ly=0,
    dt=1,
    T=1000,
    blob_shape=BlobShapeImpl(BlobShapeEnum.exp, BlobShapeEnum.gaussian),
    t_drain=2,
    periodic_y=False,
    num_blobs=10000,
    blob_factory=bf,
    one_dimensional=True,
)

ds = tmp.make_realization(speed_up=True, error=1e-10)
x = np.linspace(0, 10, 20)
t_p = 1
t_w = 1 / 10
amp = 1
v_p = 1.0
t_loss = 2.0
t_d = t_loss * t_p / (t_loss + t_p)

analytical_profile = t_d / t_w * amp * np.exp(-x / (v_p * t_loss))

ds.n.isel(y=0).mean(dim=("t")).plot(label="blob_model")
plt.yscale("log")
plt.xlabel("x")
plt.ylabel("Time average")
plt.plot(x, analytical_profile, label="analytical solution")
plt.legend()
plt.show()
