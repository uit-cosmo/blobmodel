from blobmodel import Model
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import pytest

if not os.path.isfile('./test_analytical.nc'):
    tmp = Model(Nx=100, Ny=1, Lx=10, Ly=0, dt=1, T=1000, blob_shape='exp', t_drain=2, periodic_y=False)
    tmp.sample_blobs(num_blobs=10000,A_dist='deg',W_dist='deg', vx_dist='deg',vy_dist='zeros')
    tmp.integrate(file_name='test_analytical.nc',speed_up=True, truncation_Lx = 1)


def test_convergence_to_analytical_solution():
    ds = xr.open_dataset('test_analytical.nc')
    model_profile = ds.n.isel(y=0).mean(dim=('t'))

    x = np.linspace(0,10, 100)
    t_p = 1
    t_w = 1/10
    amp = 1
    v_p = 1.0
    t_loss = 2.0
    t_d = t_loss*t_p/(t_loss+t_p)

    analytical_profile = t_d/t_w * amp * np.exp(-x/(v_p*t_loss))

    error = np.mean(abs(model_profile.values - analytical_profile))

    assert error < 0.1, "Numerical error too big"

    
test_convergence_to_analytical_solution()


