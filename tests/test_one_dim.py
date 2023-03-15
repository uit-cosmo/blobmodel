from blobmodel import Model, DefaultBlobFactory, BlobShapeImpl
import xarray as xr
import numpy as np


# use DefaultBlobFactory to define distribution functions fo random variables
bf = DefaultBlobFactory(A_dist="deg", wx_dist="deg", vx_dist="deg", vy_dist="zeros")

one_dim_model = Model(
    Nx=100,
    Ny=1,
    Lx=10,
    Ly=0,
    dt=1,
    T=1000,
    blob_shape=BlobShapeImpl("exp"),
    t_drain=2,
    periodic_y=False,
    num_blobs=10000,
    blob_factory=bf,
    one_dimensional=True,
)


def test_one_dim():
    ds = one_dim_model.make_realization(speed_up=True, error=1e-2)
    model_profile = ds.n.isel(y=0).mean(dim=("t"))

    x = np.linspace(0, 10, 100)
    t_p = 1
    t_w = 1 / 10
    amp = 1
    v_p = 1.0
    t_loss = 2.0
    t_d = t_loss * t_p / (t_loss + t_p)

    analytical_profile = t_d / t_w * amp * np.exp(-x / (v_p * t_loss))

    error = np.mean(abs(model_profile.values - analytical_profile))

    assert error < 0.1, "Numerical error too big"


test_one_dim()
