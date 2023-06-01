from blobmodel import Blob, BlobShapeImpl
import numpy as np

blob = Blob(
    blob_id=0,
    blob_shape=BlobShapeImpl("gauss"),
    amplitude=1,
    width_prop=1,
    width_perp=1,
    velocity_x=1,
    velocity_y=5,
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

    x = np.arange(0, 10, 0.1)
    y = 0

    mesh_x, mesh_y = np.meshgrid(x, y)
    blob_values = blob_sp.discretize_blob(
        x=mesh_x, y=mesh_y, t=0, periodic_y=True, Ly=10
    )

    expected_values = 1 / np.pi * np.exp(-(mesh_x**2)) * np.exp(-(4**2))
    error = np.max(abs(expected_values - blob_values))

    assert error < 1e-10, "Numerical error too big"


def test_negative_radial_velocity():
    vx = -1
    blob_sp = Blob(
        blob_id=0,
        blob_shape=BlobShapeImpl("gauss"),
        amplitude=1,
        width_prop=1,
        width_perp=1,
        velocity_x=vx,
        velocity_y=1,
        pos_x=0,
        pos_y=6,
        t_init=0,
        t_drain=10**100,
    )

    x = np.arange(-5, 5, 0.1)
    y = 0
    t = 2

    mesh_x, mesh_y = np.meshgrid(x, y)
    blob_values = blob_sp.discretize_blob(
        x=mesh_x, y=mesh_y, t=t, periodic_y=True, Ly=10
    )

    # The exact analytical expression for the expected values is a bit cumbersome, thus we just check
    # that the shape is correct
    maxx = np.max(blob_values)
    expected_values = maxx * np.exp(-((mesh_x - vx * t) ** 2))
    error = np.max(abs(expected_values - blob_values))

    assert error < 1e-10, "Numerical error too big"


def test_theta_0():
    blob_sp = Blob(
        blob_id=0,
        blob_shape=BlobShapeImpl("gauss"),
        amplitude=1,
        width_prop=1,
        width_perp=1,
        velocity_x=1,
        velocity_y=0,
        pos_x=0,
        pos_y=6,
        t_init=0,
        t_drain=10**100,
    )

    x = np.arange(0, 10, 0.1)
    y = 0

    t = np.array(2)
    mesh_x, mesh_y = np.meshgrid(x, y)
    blob_values = blob_sp.discretize_blob(
        x=mesh_x, y=mesh_y, t=t, periodic_y=True, Ly=1
    )

    # The exact analytical expression for the expected values is a bit cumbersome, thus we just check
    # that the shape is correct
    maxx = np.max(blob_values)
    expected_values = maxx * np.exp(-((mesh_x - 2) ** 2))
    error = np.max(abs(expected_values - blob_values))

    assert error < 1e-10, "Numerical error too big"


def test_kwargs():
    from unittest.mock import MagicMock

    mock_ps = BlobShapeImpl("2-exp", "2-exp")
    mock_ps.get_pulse_shape_prop = MagicMock()

    blob_sp = Blob(
        blob_id=0,
        blob_shape=mock_ps,
        amplitude=1,
        width_prop=1,
        width_perp=1,
        velocity_x=1,
        velocity_y=0,
        pos_x=0,
        pos_y=0,
        t_init=0,
        t_drain=10**100,
        prop_shape_parameters={"lam": 0.2},
        perp_shape_parameters={"lam": 0.8},
    )

    x = 0
    y = 0

    mesh_x, mesh_y = np.meshgrid(x, y)
    blob_sp.discretize_blob(x=mesh_x, y=mesh_y, t=0, periodic_y=True, Ly=10)

    mock_ps.get_pulse_shape_prop.assert_called_with([[0]], lam=0.2)
