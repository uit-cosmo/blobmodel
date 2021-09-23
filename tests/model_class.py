from blobmodel import Model, Blob

tmp = Model(Nx=200, Ny=3, Lx=10, Ly=10, dt=0.1, T=100, blob_shape='exp', t_drain=2)

#tmp.sample_blobs(num_blobs=100, vy_scale=0.001)

tmp.sample_blobs(num_blobs=1000, A_dist='deg',W_dist='deg', vx_dist='deg',vy_dist='zeros')

tmp.integrate(file_name='fast1.nc' ,speed_up=True, truncation_Lx = 1)

#tmp.show_model(interval=100, save = False)