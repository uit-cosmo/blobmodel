from blobmodel import Model, Blob

tmp = Model(Nx=200, Ny=100, Lx=0, Ly=10, dt=0.1, T=10, blob_shape='gauss', t_drain=2)

#tmp.sample_blobs(num_blobs=1000, A_dist='deg',W_dist='deg', vx_dist='deg',vy_dist='zeros')

tmp.sample_blobs(num_blobs=100)

#tmp.integrate(file_name='2d_blobs.nc' ,speed_up=False, truncation_Lx = 1)

tmp.show_model(interval=100)#), save = False)