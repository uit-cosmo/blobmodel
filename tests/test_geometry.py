import numpy as np
import pytest
from blobmodel import Blob, BlobFactory, BlobShapeImpl, Geometry, Model


@pytest.mark.parametrize(
    "N, L",
    [(10, 10.0), (100, 1.0), (98, 0.7), (1000, 0.1), (7, 2.3), (1, 5.0)],
)
def test_grid_has_exactly_n_points(N, L):
    """np.linspace grids always have exactly N points with spacing L/N,
    unlike np.arange with a float step (the classic off-by-one hazard)."""
    geo = Geometry(Nx=N, Ny=N, Lx=L, Ly=L, dt=0.1, T=1, t_init=0, periodic_y=False)
    for coord in (geo.x, geo.y):
        assert coord.size == N
        assert coord[0] == 0
        np.testing.assert_allclose(np.diff(coord), L / N, rtol=1e-12)
        assert coord[-1] < L


def test_geometry_does_not_store_meshgrids():
    """Geometry only stores 1D coordinate arrays; the full (Ny, Nx, Nt)
    meshgrids are never materialized (they held 3x the density field)."""
    geo = Geometry(Nx=5, Ny=4, Lx=1, Ly=1, dt=0.1, T=1, t_init=0, periodic_y=False)
    assert not hasattr(geo, "x_matrix")
    assert not hasattr(geo, "y_matrix")
    assert not hasattr(geo, "t_matrix")
    assert geo.x.shape == (5,)
    assert geo.y.shape == (4,)
    assert geo.t.shape == (10,)


@pytest.mark.parametrize("periodic_y", [False, True])
@pytest.mark.parametrize("theta", [None, 0.0, 0.7])
def test_discretize_blob_broadcast_matches_meshgrid(periodic_y, theta):
    """discretize_blob gives identical results for full meshgrid inputs and
    for broadcastable 1D coordinate arrays (as the Model now passes)."""
    geo = Geometry(Nx=8, Ny=6, Lx=4, Ly=3, dt=0.1, T=1, t_init=0, periodic_y=periodic_y)
    blob = Blob(
        blob_id=0,
        blob_shape=BlobShapeImpl(),
        amplitude=1.3,
        width_p=0.5,
        width_s=0.8,
        v_x=1.0,
        v_y=0.4,
        pos_x0=0.0,
        pos_y0=1.0,
        t_init=0.0,
        t_drain=10.0,
        theta=theta,
        blob_alignment=True,
    )
    x_mesh, y_mesh, t_mesh = np.meshgrid(geo.x, geo.y, geo.t)
    from_meshgrid = blob.discretize_blob(
        x=x_mesh, y=y_mesh, t=t_mesh, Ly=geo.Ly, periodic_y=periodic_y
    )
    from_broadcast = blob.discretize_blob(
        x=geo.x[np.newaxis, :, np.newaxis],
        y=geo.y[:, np.newaxis, np.newaxis],
        t=geo.t[np.newaxis, np.newaxis, :],
        Ly=geo.Ly,
        periodic_y=periodic_y,
    )
    assert from_broadcast.shape == (geo.Ny, geo.Nx, geo.t.size)
    np.testing.assert_array_equal(from_broadcast, from_meshgrid)


def test_discretize_blob_broadcast_with_array_t_drain():
    """Broadcast coordinates also work when t_drain is an array of length Nx."""
    geo = Geometry(Nx=8, Ny=6, Lx=4, Ly=3, dt=0.1, T=1, t_init=0, periodic_y=False)
    blob = Blob(
        blob_id=0,
        blob_shape=BlobShapeImpl(),
        amplitude=1.0,
        width_p=0.5,
        width_s=0.8,
        v_x=1.0,
        v_y=0.0,
        pos_x0=0.0,
        pos_y0=1.0,
        t_init=0.0,
        t_drain=np.linspace(1, 2, geo.Nx),
    )
    x_mesh, y_mesh, t_mesh = np.meshgrid(geo.x, geo.y, geo.t)
    from_meshgrid = blob.discretize_blob(
        x=x_mesh, y=y_mesh, t=t_mesh, Ly=geo.Ly, periodic_y=False
    )
    from_broadcast = blob.discretize_blob(
        x=geo.x[np.newaxis, :, np.newaxis],
        y=geo.y[:, np.newaxis, np.newaxis],
        t=geo.t[np.newaxis, np.newaxis, :],
        Ly=geo.Ly,
        periodic_y=False,
    )
    assert from_broadcast.shape == (geo.Ny, geo.Nx, geo.t.size)
    np.testing.assert_array_equal(from_broadcast, from_meshgrid)


class _ListBlobFactory(BlobFactory):
    """Returns a fixed list of blobs, ignoring all sample_blobs arguments."""

    def __init__(self, blobs):
        self._blobs = blobs

    def sample_blobs(self, Ly, T, num_blobs, blob_shape):
        return self._blobs

    def is_one_dimensional(self):
        return False


def _blob(**kwargs):
    params = dict(
        blob_id=0,
        blob_shape=BlobShapeImpl(),
        amplitude=1.0,
        width_p=0.5,
        width_s=0.5,
        v_x=1.0,
        v_y=0.0,
        pos_x0=0.0,
        pos_y0=0.0,
        t_init=0.0,
        t_drain=1e10,
    )
    params.update(kwargs)
    return Blob(**params)


def test_geometry_defaults_match_model_defaults():
    """Geometry() carries the historical Model grid defaults, so Model() keeps
    working without any grid parameters."""
    geo = Geometry()
    assert (geo.Nx, geo.Ny, geo.Lx, geo.Ly) == (100, 100, 10, 10)
    assert (geo.dt, geo.T, geo.t_init) == (0.1, 10, 0)
    assert geo.periodic_y is False
    assert (geo.x0, geo.y0) == (0, 0)


@pytest.mark.parametrize("x0, y0", [(-5.0, -5.0), (2.5, 0.0), (0.0, 7.0)])
def test_domain_origin_offsets(x0, y0):
    """x0/y0 shift the coordinate arrays without changing spacing or size."""
    geo = Geometry(Nx=10, Ny=10, Lx=10, Ly=10, dt=0.1, T=1, x0=x0, y0=y0)
    assert geo.x[0] == pytest.approx(x0)
    assert geo.y[0] == pytest.approx(y0)
    reference = Geometry(Nx=10, Ny=10, Lx=10, Ly=10, dt=0.1, T=1)
    np.testing.assert_allclose(geo.x, reference.x + x0)
    np.testing.assert_allclose(geo.y, reference.y + y0)


def test_one_dimensional_geometry_y_is_y0():
    """For Ly == 0 the single y point sits at the domain origin."""
    geo = Geometry(Ny=1, Ly=0, y0=3.0)
    np.testing.assert_array_equal(geo.y, [3.0])


def test_from_arrays_roundtrip():
    """from_arrays derives the invariants a directly-constructed Geometry has."""
    reference = Geometry(
        Nx=16, Ny=8, Lx=4, Ly=2, dt=0.5, T=5, t_init=-2.5, x0=-2.0, y0=1.0
    )
    geo = Geometry.from_arrays(reference.x, reference.y, reference.t, periodic_y=True)
    assert geo.Nx == reference.Nx
    assert geo.Ny == reference.Ny
    assert geo.Lx == pytest.approx(reference.Lx)
    assert geo.Ly == pytest.approx(reference.Ly)
    assert geo.dt == pytest.approx(reference.dt)
    assert geo.t_init == pytest.approx(reference.t_init)
    assert geo.x0 == pytest.approx(reference.x0)
    assert geo.y0 == pytest.approx(reference.y0)
    assert geo.periodic_y is True
    np.testing.assert_array_equal(geo.x, reference.x)
    np.testing.assert_array_equal(geo.y, reference.y)
    np.testing.assert_array_equal(geo.t, reference.t)


def test_from_arrays_single_point_y_gives_1d_geometry():
    geo = Geometry.from_arrays(
        np.linspace(0, 1, 10), np.array([0.0]), np.arange(0, 1, 0.1)
    )
    assert geo.Ny == 1
    assert geo.Ly == 0


def test_from_arrays_rejects_bad_input():
    x = np.linspace(0, 1, 10)
    t = np.arange(0, 1, 0.1)
    with pytest.raises(ValueError, match="uniformly spaced"):
        Geometry.from_arrays(x, np.array([0.0, 0.1, 0.5]), t)
    with pytest.raises(ValueError, match="1D"):
        Geometry.from_arrays(np.zeros((2, 2)), x, t)
    with pytest.raises(ValueError, match="1D"):
        Geometry.from_arrays(x, x, np.array([]))
    # descending arrays are uniformly spaced but would give negative domain
    # lengths, breaking the speed_up truncation windows
    with pytest.raises(ValueError, match="strictly increasing"):
        Geometry.from_arrays(x[::-1], np.array([0.0]), t)
    with pytest.raises(ValueError, match="strictly increasing"):
        Geometry.from_arrays(x, x, t[::-1])


def test_model_accepts_geometry_and_exposes_it():
    """A user-built Geometry is used as-is and readable via model.geometry."""
    geo = Geometry(Nx=4, Ny=4, Lx=2, Ly=2, dt=0.5, T=1)
    model = Model(geometry=geo, num_blobs=1, verbose=False)
    assert model.geometry is geo
    ds = model.make_realization()
    np.testing.assert_array_equal(ds.x, geo.x)
    np.testing.assert_array_equal(ds.y, geo.y)
    np.testing.assert_array_equal(ds.t, geo.t)


def test_model_geometry_is_read_only():
    model = Model(num_blobs=1, verbose=False)
    with pytest.raises(AttributeError):
        model.geometry = Geometry()


def test_model_rejects_non_geometry():
    with pytest.raises(TypeError, match="geometry"):
        Model(geometry="not a geometry")


def test_offset_domain_is_translation_invariant():
    """A realization on a domain offset by (x0, y0) with blobs shifted by the
    same amount equals the unshifted reference realization -- including the
    periodic-y wrapping, which must wrap into [y0, y0 + Ly)."""
    x0, y0 = -5.0, 10.0
    kwargs = dict(Nx=8, Ny=6, Lx=4, Ly=3, dt=0.1, T=2, periodic_y=True)
    blob_kwargs = dict(v_x=1.0, v_y=2.0, pos_x0=1.0, pos_y0=2.5, t_init=0.0)

    reference = Model(
        geometry=Geometry(**kwargs),
        blob_factory=_ListBlobFactory([_blob(**blob_kwargs)]),
        verbose=False,
    ).make_realization()

    shifted_blob_kwargs = dict(
        blob_kwargs,
        pos_x0=blob_kwargs["pos_x0"] + x0,
        pos_y0=blob_kwargs["pos_y0"] + y0,
    )
    shifted = Model(
        geometry=Geometry(**kwargs, x0=x0, y0=y0),
        blob_factory=_ListBlobFactory([_blob(**shifted_blob_kwargs)]),
        verbose=False,
    ).make_realization()

    np.testing.assert_allclose(shifted.n.values, reference.n.values, atol=1e-12)
