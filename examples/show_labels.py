from blobmodel import Model, show_model
import numpy as np

# here you can define your custom parameter distributions

bm = Model(
    Nx=100,
    Ny=100,
    Lx=20,
    Ly=20,
    dt=0.1,
    T=20,
    periodic_y=True,
    blob_shape="gauss",
    num_blobs=10,
    t_drain=1e10,
    labels="individual",
)

# create data
ds = bm.make_realization(speed_up=True, error=1e-2)

import matplotlib.pyplot as plt

ds.n.isel(t=-1).plot()
plt.figure()
ds.blob_labels.isel(t=-1).plot()

plt.show()
