from blobmodel import Model

bm = Model(Nx=100, Ny=50, Lx=10, Ly=10, dt=0.1, T=20, blob_shape="exp", num_blobs=1000)

# save data as nc file
# use speedup option with blob truncated after propagating length 2*Lx
bm.integrate(file_name="example.nc", speed_up=True, truncation_Lx=2)
