from model import Model, Blob

tmp = Model(Nx=100, Ny=100, Lx=100, Ly=100, dt=0.01, T=1)

tmp.sample_blobs()

print(tmp.Nx)

blob = Blob(1,1,1,1,1,1,1,1,1)

print(blob)