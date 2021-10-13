from blobmodel import Model, Geometry

geo = Geometry(Nx=100, Ny=100, Lx=10, Ly=10, dt=0.1, T=10, periodic_y=False,)
bm = Model(geometry=geo, blob_shape="exp", num_blobs=1000)

# save data as nc file
# use speedup option with blob truncated after propagating length 2*Lx
bm.integrate(file_name="example.nc", speed_up=True, truncation_Lx=2)
