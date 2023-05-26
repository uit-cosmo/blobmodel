from blobmodel import Blob, BlobShapeImpl
import numpy as np

blob_sp = Blob(
    blob_id=0,
    blob_shape=BlobShapeImpl("gauss"),
    amplitude=1,
    width_prop=1,
    width_perp=1,
    velocity_x=1,
    velocity_y=1,
    pos_x=0,
    pos_y=6,
    t_init=0,
    t_drain=10**100,
)

x = 0
y = 0
times = np.arange(1, 5, 0.01)

mesh_x, mesh_y, mesh_t = np.meshgrid(x, y, times)
blob_values = blob_sp.discretize_blob(
    x=mesh_x, y=mesh_y, t=mesh_t, periodic_y=True, Ly=10
)
