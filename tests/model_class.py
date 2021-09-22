from blobmodel import Model, Blob

tmp = Model(Nx=100, Ny=50, Lx=10, Ly=10, dt=0.1, T=20, blob_shape='gauss')

tmp.sample_blobs(num_blobs=50)

#tmp.integrate()

tmp.show_model(interval=100)