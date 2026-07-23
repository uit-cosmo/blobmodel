from blobmodel import Geometry, Model, show_model

# Example of a simple two-dimensional realization and animation.

bm = Model(
    geometry=Geometry(
        Nx=100, Ny=100, Lx=10, Ly=10, dt=0.1, T=20, periodic_y=True, t_init=10
    ),
    num_blobs=100,
)

# Make a realization and store it in a DataSet object
ds = bm.make_realization(truncation_error=1e-2)
# show animation and save as gif
show_model(dataset=ds, interval=100, gif_name="2d_animation.gif", fps=10)
