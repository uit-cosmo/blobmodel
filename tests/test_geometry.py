import numpy as np
import pytest
from blobmodel import Blob, BlobShapeImpl, Geometry


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
