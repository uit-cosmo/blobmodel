from blobmodel import Model, DefaultBlobFactory
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os

# use DefaultBlobFactory to define distribution functions fo random variables
bf = DefaultBlobFactory(A_dist="deg", wx_dist="deg", vx_dist="deg", vy_dist="zeros")

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

ds = tmp.make_realization(file_name="profile_comparison.nc", speed_up=False, error=1e-4)


def plot_convergence_to_analytical_solution(ds):
    x = np.linspace(0, 10, 100)
    t_p = 1
    t_w = 1 / 10
    amp = 1
    v_p = 1.0
    t_loss = 2.0
    t_d = t_loss * t_p / (t_loss + t_p)

    analytical_profile = (
        1 / np.sqrt(np.pi) * t_d / t_w * amp * np.exp(-x / (v_p * t_loss))
    )

    ds.n.isel(y=0).mean(dim=("t")).plot(label="blob_model")
    plt.yscale("log")
    plt.plot(x, analytical_profile, label="analytical solution")
    plt.legend()
    plt.show()


plot_convergence_to_analytical_solution(ds)
