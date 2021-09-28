from blobmodel import Model

bm = Model(Nx=100, Ny=100, Lx=10, Ly=10, dt=0.1, T=20, blob_shape='gauss')
#bm.sample_blobs(num_blobs=1000, A_dist='deg',W_dist='deg', vx_dist='deg',vy_dist='zeros')

bm.sample_blobs(num_blobs=1000, W_dist="deg", vy_dist="zeros", vx_dist="deg")


bm.integrate(file_name='2d_blobs.nc' ,speed_up=True, truncation_Lx = 1)

#bm.show_model(interval=100)#), save = False)

# show animation and save as gif
bm.show_model(interval=100, save = True, gif_name = 'example.gif', fps = 10)
