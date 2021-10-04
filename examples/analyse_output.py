from blobmodel import Model
import xarray as xr
import matplotlib.pyplot as plt
import os

if not os.path.isfile('./example.nc'):
    bm = Model(Nx=100, Ny=50, Lx=10, Ly=10, dt=0.1, T=20, blob_shape='exp')
    bm.sample_blobs(num_blobs=1000)
    bm.integrate(file_name='example.nc', speed_up=True, truncation_Lx=2)

# use xarray to open output
ds = xr.open_dataset('example.nc')

# use xarray syntax for analysing output
# for example
ds['n'].isel(y=0).mean(dim=('t')).plot()
plt.show()
