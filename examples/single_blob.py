from blobmodel import (
    Model,
    show_model,
)

# Example of a realization with a single blob

bm = Model(
    Nx=10,
    Ny=10,
    Lx=10,
    Ly=10,
    dt=0.1,
    T=10,
    periodic_y=True,
    num_blobs=1,
    t_drain=1e10,
)

# create data
ds = bm.make_realization(speed_up=True, error=1e-2)

# show animation and save as gif
show_model(dataset=ds, interval=100, gif_name="example.gif", fps=10)
