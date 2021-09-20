from model import Model, Blob
import numpy as np
import matplotlib.pyplot as plt

tmp = Model(Nx=100, Ny=100, Lx=10, Ly=10, dt=0.1, T=20)

# blob = Blob(1, 'gauss', 1,1,1,1,1,1,1,1)

# x = np.linspace(0, 10, 100)
# y = np.linspace(0, 10, 100).reshape(-1, 1)
# T = np.linspace(0, 10, 100)

# tmp = blob.discretize_blob(x,y,3)
# plt.contourf(tmp)
# plt.show()


tmp.sample_blobs(num_blobs=200)


tmp.show_model(interval=100)