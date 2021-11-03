from blobmodel import Model, show_model
import numpy as np

# here you can define your custom parameter distributions

bm = Model(
    Nx=100,
    Ny=100,
    Lx=10,
    Ly=10,
    dt=0.1,
    T=20,
    periodic_y=True,
    blob_shape="gauss",
    num_blobs=100,
    t_drain=1e10,
)

# create data
ds = bm.make_realization(speed_up=True, error=1e-2)
# show animation and save as gif
show_model(ds=ds, interval=100, save=True, gif_name="example.gif", fps=10)
