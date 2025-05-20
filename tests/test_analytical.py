from blobmodel import (
    Model,
    DefaultBlobFactory,
    DistributionEnum,
    BlobShapeImpl,
    BlobShapeEnum,
)
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


# use DefaultBlobFactory to define distribution functions fo random variables
bf = DefaultBlobFactory(A_dist=DistributionEnum.deg, vy_dist=DistributionEnum.zeros)

t_drain = 2
Nx, Lx = 10, 10
tmp = Model(
    Nx=Nx,
    Ny=1,
    Lx=Lx,
    Ly=0,
    dt=1,
    T=1000,
    blob_shape=BlobShapeImpl(BlobShapeEnum.exp, BlobShapeEnum.gaussian),
    t_drain=t_drain,
    periodic_y=False,
    num_blobs=10000,
    blob_factory=bf,
    one_dimensional=True,
)


def test_convergence_to_analytical_solution():
    """
    Checks that the mean value of a one-dimensional realization with constant velocities agrees with the
    analytical derived results. See O. E. Garcia, et al.; Phys. Plasmas 1 May 2016; 23 (5): 052308. https://doi.org/10.1063/1.4951016
    """
    ds = tmp.make_realization(speed_up=True, error=1e-10)
    model_profile = ds.n.isel(y=0).mean(dim="t")

    x = np.arange(0, Lx, Lx / Nx)
    t_p = 1  # vx/ell
    t_w = 1 / 10
    amp = 1
    v_p = 1.0
    t_d = t_drain * t_p / (t_drain + t_p)

    analytical_profile = t_d / t_w * amp * np.exp(-x / (v_p * t_drain))
    error = np.mean(abs(model_profile.values - analytical_profile))

    assert error < 0.1, "Numerical error too big"
