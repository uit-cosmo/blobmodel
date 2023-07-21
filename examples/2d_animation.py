from blobmodel import Model, show_model

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
show_model(dataset=ds, interval=100, gif_name="2d_animation.gif", fps=10)
