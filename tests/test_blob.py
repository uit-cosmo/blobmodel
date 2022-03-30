from blobmodel import Blob
import numpy as np

blob = Blob(
    blob_id=0,
    blob_shape="gauss",
    amplitude=1,
    width_prop=1,
    width_perp=1,
    v_x=1,
    v_y=5,
    pos_x=0,
    pos_y=0,
    t_init=0,
    t_drain=10**10,
)


def test_initial_blob():
    x = np.arange(0, 10, 0.1)
    y = np.array([0, 1])

    mesh_x, mesh_y = np.meshgrid(x, y)
    blob_values = blob.discretize_blob(x=mesh_x, y=mesh_y, t=0, periodic_y=False, Ly=10)

    expected_values = 1 / np.pi * np.exp(-(mesh_x**2)) * np.exp(-(mesh_y**2))
    error = np.max(abs(expected_values - blob_values))

    assert error < 1e-10, "Numerical error too big"


def test_periodicity():
    x = np.arange(0, 10, 0.1)
    y = np.arange(0, 10, 1)
    t = np.array(2)
    mesh_x, mesh_y, mesh_t = np.meshgrid(x, y, t)
    blob_values = blob.discretize_blob(
        x=mesh_x, y=mesh_y, t=mesh_t, periodic_y=True, Ly=10
    )

    expected_values = 1 / np.pi * np.exp(-((mesh_x - 2) ** 2)) * np.exp(
        -(mesh_y**2)
    ) + 1 / np.pi * np.exp(-((mesh_x - 2) ** 2)) * np.exp(-((mesh_y - 10) ** 2))
    error = np.max(abs(expected_values - blob_values))

    assert error < 1e-10, "Numerical error too big"


def test_single_point():
    blob_sp = Blob(
        blob_id=0,
        blob_shape="gauss",
        amplitude=1,
        width_prop=1,
        width_perp=1,
        v_x=1,
        v_y=1,
        pos_x=0,
        pos_y=6,
        t_init=0,
        t_drain=10**100,
    )

    x = np.arange(0, 10, 0.1)
    y = 0

    mesh_x, mesh_y = np.meshgrid(x, y)
    blob_values = blob_sp.discretize_blob(
        x=mesh_x, y=mesh_y, t=0, periodic_y=True, Ly=10
    )

    expected_values = 1 / np.pi * np.exp(-(mesh_x**2)) * np.exp(-(4**2))
    error = np.max(abs(expected_values - blob_values))

    assert error < 1e-10, "Numerical error too big"


test_initial_blob()
test_periodicity()
test_single_point()
