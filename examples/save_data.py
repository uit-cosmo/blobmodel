from blobmodel import Model

bm = Model(
    Nx=100,
    Ny=100,
    Lx=10,
    Ly=10,
    dt=0.1,
    T=10,
    periodic_y=False,
    blob_shape="gauss",
    num_blobs=100,
)

# save data as nc file
# use speedup option with blob truncated after propagating length 2*Lx
bm.make_realization(file_name="example.nc", speed_up=True, error=0.1)
