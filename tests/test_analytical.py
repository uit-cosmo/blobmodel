from blobmodel import (
    Model,
    DefaultBlobFactory,
    DistributionEnum,
    BlobShapeImpl,
    BlobShapeEnum,
)
import xarray as xr
import numpy as np


# use DefaultBlobFactory to define distribution functions fo random variables
bf = DefaultBlobFactory(A_dist=DistributionEnum.deg, vy_dist=DistributionEnum.zeros)

t_loss = 2.0

tmp = Model(
    Nx=100,
    Ny=1,
    Lx=10,
    Ly=0,
    dt=0.1,
    T=1000,
    t_drain=t_loss,
    blob_shape=BlobShapeImpl(
        BlobShapeEnum.exp, BlobShapeEnum.gaussian
    ),  # Analytical form only applies to exp shape
    periodic_y=False,
    num_blobs=10000,
    blob_factory=bf,
    t_init=10,
    one_dimensional=True,
)

tmp.make_realization(file_name="test_analytical.nc", speed_up=True, error=1e-2)


def test_convergence_to_analytical_solution():
    """
    Checks that the mean value of a one-dimensional realization with constant velocities agrees with the
    analytical derived results. See O. E. Garcia, et al.; Phys. Plasmas 1 May 2016; 23 (5): 052308. https://doi.org/10.1063/1.4951016
    """
    ds = xr.open_dataset("test_analytical.nc")
    model_profile = ds.n.isel(y=0).mean(dim="t")

    x = np.linspace(0, 10, 100)
    t_p = 1  # vx/ell
    t_w = 1 / 10
    amp = 1
    v_p = 1.0
    t_d = t_loss * t_p / (t_loss + t_p)

    analytical_profile = t_d / t_w * amp * np.exp(-x / (v_p * t_loss))

    error = np.mean(abs(model_profile.values - analytical_profile))

    assert error < 0.1, "Numerical error too big"
