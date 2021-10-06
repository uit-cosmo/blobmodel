from blobmodel import Blob
import numpy as np

blob = Blob(
    id=0,
    blob_shape="gauss",
    amplitude=1,
    width_x=1,
    width_y=1,
    v_x=1,
    v_y=1,
    pos_x=0,
    pos_y=0,
    t_init=0,
    t_drain=10**100)


def test_initial_blob():
    x = np.arange(0, 10, 0.1)
    y = np.array([0, 1])

    mesh_x, mesh_y = np.meshgrid(x, y)
    blob_values = blob.discretize_blob(x=mesh_x, y=mesh_y, t=0, periodic_y=False, Ly=10)

    expected_values = 1/np.pi * np.exp(-mesh_x**2/2) * np.exp(-mesh_y**2/2)
    error = np.mean(abs(expected_values - blob_values))

    assert error < 0.1, "Numerical error too big"


def test_periodicity():
    x = np.arange(0, 10, 0.1)
    y = np.array([0, 1])

    mesh_x, mesh_y = np.meshgrid(x, y)
    blob_values = blob.discretize_blob(x=mesh_x, y=mesh_y, t=2, periodic_y=True, Ly=1)

    expected_values = 1/np.pi * np.exp(-(mesh_x - 2)**2/2) * np.exp(-mesh_y**2/2)
    error = np.mean(abs(expected_values - blob_values))

    assert error < 0.1, "Numerical error too big"


test_initial_blob()
test_periodicity()
