from blobmodel import Model, show_model

# Example of a simple two-dimensional realization and animation.

bm = Model(
    Nx=100,
    Ny=100,
    Lx=10,
    Ly=10,
    dt=0.1,
    T=20,
    periodic_y=True,
    num_blobs=100,
    t_drain=1e10,
    t_init=10,
)

# Make a realization and store it in a DataSet object
ds = bm.make_realization(speed_up=True, error=1e-2)
# show animation and save as gif
show_model(dataset=ds, interval=100, gif_name="2d_animation.gif", fps=10)
