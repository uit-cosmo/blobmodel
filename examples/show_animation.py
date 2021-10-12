from blobmodel import Model, show_model

bm = Model(Nx=100, Ny=50, Lx=10, Ly=10, dt=0.1, T=20, blob_shape="gauss")

bm.sample_blobs(num_blobs=1000)

# create data
ds = bm.integrate(speed_up=True, truncation_Lx=2)

# show animation and save as gif
show_model(ds=ds, interval=100, save=True, gif_name="example.gif", fps=10)
