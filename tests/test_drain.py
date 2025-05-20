from blobmodel import Blob, BlobShapeImpl
import numpy as np


def test_high_t_drain():
    """
    Checks that high t drain runs without errors.
    """
    blob_sp = Blob(
        blob_id=0,
        amplitude=1,
        width_prop=1,
        width_perp=1,
        blob_shape=BlobShapeImpl(),
        v_x=1,
        v_y=1,
        pos_x=0,
        pos_y=6,
        t_init=0,
        t_drain=10**100,
    )

    x = 0
    y = 0
    times = np.arange(1, 5, 0.01)

    mesh_x, mesh_y, mesh_t = np.meshgrid(x, y, times)
    blob_sp.discretize_blob(x=mesh_x, y=mesh_y, t=mesh_t, periodic_y=False, Ly=10)
