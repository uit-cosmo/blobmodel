from blobmodel import Model, DefaultBlobFactory, Blob, BlobShapeImpl
import numpy as np
from unittest.mock import MagicMock

blob = Blob(
    blob_id=0,
    blob_shape=BlobShapeImpl("gauss"),
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


def test_blob_non_alignment():
    # Dead pixels have already been preprocessed and have an array of nans at their site
    blob = Blob(
        blob_id=0,
        blob_shape=BlobShapeImpl("exp", "exp"),
        amplitude=1,
        width_prop=1,
        width_perp=1,
        v_x=1,
        v_y=1,
        pos_x=0,
        pos_y=0,
        t_init=0,
        t_drain=10**10,
        blob_alignment=False,
    )
    x = np.arange(0, 10, 0.1)
    y = np.array([0, 1])

    mesh_x, mesh_y = np.meshgrid(x, y)
    mock = MagicMock(return_value=mesh_x)
    blob.blob_shape.get_blob_shape_prop = mock
    blob.discretize_blob(x=mesh_x, y=mesh_y, t=0, periodic_y=False, Ly=10)

    np.testing.assert_array_equal(mesh_x, mock.call_args[0][0])


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


def test_negative_radial_velocity():
    vx = -1
    blob_sp = Blob(
        blob_id=0,
        blob_shape=BlobShapeImpl("gauss"),
        amplitude=1,
        width_prop=1,
        width_perp=1,
        v_x=vx,
        v_y=1,
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
        v_x=1,
        v_y=0,
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
    mock_ps.get_blob_shape_prop = MagicMock()

    blob_sp = Blob(
        blob_id=0,
        blob_shape=mock_ps,
        amplitude=1,
        width_prop=1,
        width_perp=1,
        v_x=1,
        v_y=0,
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
    mock_ps.get_blob_shape_prop.assert_called_with([[0]], lam=0.2)


def test_get_blobs():
    bf = DefaultBlobFactory(A_dist="deg", wx_dist="deg", vx_dist="deg", vy_dist="deg")
    one_blob = Model(
        Nx=100,
        Ny=100,
        Lx=10,
        Ly=10,
        dt=1,
        T=1,
        blob_shape="exp",
        t_drain=1e10,
        periodic_y=False,
        num_blobs=3,
        blob_factory=bf,
    )
    ds = one_blob.make_realization()
    blob_list = one_blob.get_blobs()
    assert len(blob_list) == 3
