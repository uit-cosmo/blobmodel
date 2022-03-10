from blobmodel import Model, show_model, DefaultBlobFactory

bf = DefaultBlobFactory(A_dist="deg", W_dist="deg", vx_dist="deg", vy_dist="zeros")

bm = Model(
    Nx=100,
    Ny=1,
    Lx=10,
    Ly=0,
    dt=0.1,
    T=20,
    periodic_y=False,
    blob_shape="exp",
    num_blobs=100,
    t_drain=10,
    blob_factory=bf,
)

ds = bm.make_realization(speed_up=True, error=1e-2)
show_model(dataset=ds, interval=100, save=True, gif_name="example.gif", fps=10)
