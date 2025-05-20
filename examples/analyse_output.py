from blobmodel import Model, BlobShapeImpl, BlobShapeEnum
import xarray as xr
import matplotlib.pyplot as plt
import os

# Example reading data from a file stored in "example.nc" and plotting the time mean using xarray functions.


if not os.path.isfile("./example.nc"):
    bm = Model(
        Nx=100,
        Ny=100,
        Lx=10,
        Ly=10,
        dt=0.1,
        T=10,
        blob_shape=BlobShapeImpl(BlobShapeEnum.exp, BlobShapeEnum.gaussian),
        num_blobs=1000,
    )
    bm.make_realization(file_name="example.nc", speed_up=True, error=1e-2)

# use xarray to open output
ds = xr.open_dataset("example.nc")

# use xarray syntax for analysing output
# for example
ds["n"].isel(y=0).mean(dim=("t")).plot()
plt.show()
