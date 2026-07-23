from blobmodel import (
    Geometry,
    Model,
    BlobFactory,
    Blob,
    AbstractBlobShape,
    BlobShapeImpl,
    BlobShapeEnum,
)
import numpy as np
import numbers
import pytest
from typing import List

ERROR = 1e-10


class SingleBlobFactory(BlobFactory):
    """Yields one deterministic blob with fully user-specified parameters."""

    def __init__(self, **blob_kwargs) -> None:
        params = dict(
            amplitude=1.0,
            width_p=0.2,
            width_s=0.2,
            v_x=1.0,
            v_y=0.0,
            pos_x0=0.0,
            pos_y0=0.0,
            t_init=0.0,
            t_drain=1e10,
        )
        params.update(blob_kwargs)
        self._params = params

    def sample_blobs(
        self,
        Ly: float,
        T: float,
        num_blobs: int,
        blob_shape: AbstractBlobShape,
    ) -> List[Blob]:
        return [Blob(blob_id=0, blob_shape=blob_shape, **self._params)]

    def is_one_dimensional(self) -> bool:
        return True


def _make_model(
    factory: SingleBlobFactory,
    T: float = 12,
    geometry_t0: float = 0.0,
    x0: float = 0.0,
) -> Model:
    # Nx == Lx puts grid points on the integers. Only the geometry is used
    # here (no realization is computed), so the model stays cheap. geometry_t0
    # (the geometry's t_init) sets where the time axis np.arange(t0, T, dt)
    # starts; x0 sets where the x-domain [x0, x0 + Lx) starts.
    return Model(
        geometry=Geometry(
            Nx=10,
            Ny=1,
            Lx=10,
            Ly=0,
            dt=0.1,
            T=T,
            t_init=geometry_t0,
            periodic_y=False,
            x0=x0,
        ),
        blob_shape=BlobShapeImpl(BlobShapeEnum.gaussian, BlobShapeEnum.gaussian),
        num_blobs=1,
        blob_factory=factory,
        one_dimensional=True,
    )


def _single_blob(factory: SingleBlobFactory) -> Blob:
    return factory.sample_blobs(
        Ly=0,
        T=12,
        num_blobs=1,
        blob_shape=BlobShapeImpl(BlobShapeEnum.gaussian, BlobShapeEnum.gaussian),
    )[0]


def _start_stop(
    factory,
    speed_up: bool = True,
    error: float = ERROR,
    geometry_t0: float = 0.0,
    x0: float = 0.0,
):
    """Call _compute_start_stop directly; no discretization is performed."""
    model = _make_model(factory, geometry_t0=geometry_t0, x0=x0)
    blob = _single_blob(factory)
    start, stop = model._compute_start_stop(blob, speed_up, error)
    return start, stop, model, blob


def _support_indices(model: Model, blob: Blob, error: float) -> np.ndarray:
    """
    Time indices at which the blob's peak contribution inside the sampled
    grid still exceeds `error` (absolute threshold, drain ignored). Derived
    from the trajectory only -- no field discretization.

    The [start, stop) window returned by _compute_start_stop must contain all
    of these indices, otherwise speed_up truncates part of the blob.
    """
    geom = model._geometry
    pos = blob.pos_x0 + blob.v_x * (geom.t - blob.t_init)
    # distance from the blob centre to the nearest sampled grid column
    left, right = geom.x[0], geom.x[-1]
    d = np.abs(np.clip(pos, left, right) - pos)  # 0 while inside the grid
    peak = blob.amplitude / np.sqrt(np.pi) * np.exp(-((d / blob.width_p) ** 2))
    return np.where(peak >= error)[0]


CONFIGS = [
    dict(pos_x0=0.0, v_x=1.0, t_init=0.0),  # born at the inflow edge
    dict(pos_x0=5.0, v_x=1.0, t_init=0.0),  # born mid-domain
    dict(pos_x0=8.0, v_x=1.0, t_init=0.0),  # born near the outflow edge
    dict(pos_x0=5.0, v_x=1.0, t_init=3.0),  # delayed birth
    dict(pos_x0=3.0, v_x=2.0, t_init=0.0),  # faster blob
    dict(pos_x0=5.0, v_x=-1.0, t_init=0.0),  # leftward, born mid-domain
    dict(pos_x0=2.0, v_x=-1.0, t_init=0.0),  # leftward, born left of centre
    dict(pos_x0=8.0, v_x=-1.0, t_init=0.0),  # leftward, born near right edge
    dict(pos_x0=13.0, v_x=-1.0, t_init=0.0),  # born right of domain, drifts in
    dict(pos_x0=-3.0, v_x=1.0, t_init=0.0),  # born left of domain, drifts in
]


@pytest.mark.parametrize("blob_kwargs", CONFIGS)
def test_compute_start_stop_returns_integer_window(blob_kwargs):
    """
    start and stop are used as array slice bounds, so they must be genuine
    integers, ordered, and inside [0, t.size]. A float bound (division left
    outside the int()) or an inverted window (start > stop) is a bug.
    """
    start, stop, model, _ = _start_stop(SingleBlobFactory(**blob_kwargs))
    size = model._geometry.t.size

    assert isinstance(start, numbers.Integral)
    assert isinstance(stop, numbers.Integral)
    assert 0 <= start <= stop <= size


@pytest.mark.parametrize("blob_kwargs", CONFIGS)
def test_compute_start_stop_covers_blob_support(blob_kwargs):
    """
    The [start, stop) window must contain every time index at which the blob
    still contributes above `error`; otherwise speed_up silently truncates it.
    """
    start, stop, model, blob = _start_stop(SingleBlobFactory(**blob_kwargs))
    support = _support_indices(model, blob, ERROR)

    assert support.size > 0  # guard: each config really does contribute
    assert start <= support.min()
    assert stop > support.max()  # stop is an exclusive slice bound


# (geometry_t0, blob_kwargs). The time axis is np.arange(geometry_t0, T, dt),
# so t[0] != 0 and the blob t_init generally differs from it.
SHIFTED_GRID_CONFIGS = [
    (5.0, dict(pos_x0=0.0, v_x=1.0, t_init=5.0)),  # born at grid start, rightward
    (5.0, dict(pos_x0=0.0, v_x=2.0, t_init=8.0)),  # born mid-window -> start > 0
    (5.0, dict(pos_x0=5.0, v_x=-2.0, t_init=7.0)),  # leftward, exits before grid end
    (-3.0, dict(pos_x0=2.0, v_x=1.0, t_init=-3.0)),  # negative grid start
    (-3.0, dict(pos_x0=5.0, v_x=-1.0, t_init=0.0)),  # blob t_init != grid start
]


@pytest.mark.parametrize("geometry_t0, blob_kwargs", SHIFTED_GRID_CONFIGS)
def test_compute_start_stop_with_shifted_time_grid(geometry_t0, blob_kwargs):
    """
    start/stop are indices into np.arange(geometry_t0, T, dt), so they must be
    rebased by t[0], not by 0. This exercises the (blob.t_init - t0) term: a
    slip there would shift the window off the blob's support once t[0] != 0.
    """
    start, stop, model, blob = _start_stop(
        SingleBlobFactory(**blob_kwargs), geometry_t0=geometry_t0
    )
    size = model._geometry.t.size

    assert model._geometry.t[0] == pytest.approx(geometry_t0)
    assert isinstance(start, numbers.Integral)
    assert isinstance(stop, numbers.Integral)
    assert 0 <= start <= stop <= size

    support = _support_indices(model, blob, ERROR)
    assert support.size > 0
    assert start <= support.min()
    assert stop > support.max()  # stop is an exclusive slice bound


# (x0, blob_kwargs). The x-domain is [x0, x0 + Lx), so x[0] != 0 and blob
# positions are absolute coordinates that generally differ from x0.
OFFSET_X_CONFIGS = [
    (-5.0, dict(pos_x0=-5.0, v_x=1.0, t_init=0.0)),  # born at the inflow edge
    (-5.0, dict(pos_x0=0.0, v_x=1.0, t_init=0.0)),  # born mid-domain
    (-5.0, dict(pos_x0=-8.0, v_x=1.0, t_init=0.0)),  # born left of domain, drifts in
    (-5.0, dict(pos_x0=3.0, v_x=-2.0, t_init=2.0)),  # leftward, born near right edge
    (20.0, dict(pos_x0=20.0, v_x=1.0, t_init=0.0)),  # domain far from the origin
]


@pytest.mark.parametrize("x0, blob_kwargs", OFFSET_X_CONFIGS)
def test_compute_start_stop_with_offset_x_domain(x0, blob_kwargs):
    """
    The truncation window is derived from geometry.x[0] and geometry.Lx, so a
    domain-origin offset x0 must not shift the window off the blob's support
    (a stale assumption of x[0] == 0 would).
    """
    start, stop, model, blob = _start_stop(SingleBlobFactory(**blob_kwargs), x0=x0)
    size = model._geometry.t.size

    assert model._geometry.x[0] == pytest.approx(x0)
    assert isinstance(start, numbers.Integral)
    assert isinstance(stop, numbers.Integral)
    assert 0 <= start <= stop <= size

    support = _support_indices(model, blob, ERROR)
    assert support.size > 0
    assert start <= support.min()
    assert stop > support.max()  # stop is an exclusive slice bound


@pytest.mark.parametrize("x0, blob_kwargs", OFFSET_X_CONFIGS)
def test_speed_up_realization_matches_full_on_offset_domain(x0, blob_kwargs):
    """
    End-to-end guard for offset domains: with speed_up the realization must
    match the untruncated one to within the truncation error.
    """
    factory = SingleBlobFactory(**blob_kwargs)
    ds_full = _make_model(factory, x0=x0).make_realization(speed_up=False)
    ds_fast = _make_model(factory, x0=x0).make_realization(truncation_error=ERROR)
    np.testing.assert_allclose(ds_fast.n.values, ds_full.n.values, atol=10 * ERROR)


def test_make_realization_defaults_to_speed_up():
    """
    speed_up=True with truncation_error=1e-10 is the documented default
    (every surveyed downstream call passed exactly this); a silent flip back
    to False would go unnoticed by the other tests, which all pass the flag
    explicitly.
    """
    import inspect

    params = inspect.signature(Model.make_realization).parameters
    assert params["speed_up"].default is True
    assert params["truncation_error"].default == 1e-10


@pytest.mark.parametrize("speed_up, v_x", [(False, 1.0), (True, 0.0)])
def test_compute_start_stop_full_window_when_not_applicable(speed_up, v_x):
    """
    speed_up disabled, or v_x == 0 (no x-velocity to window on), must return
    the whole time axis.
    """
    start, stop, model, _ = _start_stop(
        SingleBlobFactory(pos_x0=5.0, v_x=v_x), speed_up=speed_up
    )
    assert (int(start), int(stop)) == (0, model._geometry.t.size)


@pytest.mark.parametrize("blob_kwargs", CONFIGS)
def test_compute_start_stop_is_not_wastefully_wide(blob_kwargs):
    """
    Complement to the coverage test: the window must not extend far past the
    blob's support, or speed_up buys nothing. It may exceed the support by at
    most the truncation margin on each side (plus rounding); a window that
    stayed full-width for an off-centre blob -- e.g. a wrong-signed exit
    distance -- would blow this bound.
    """
    start, stop, model, blob = _start_stop(SingleBlobFactory(**blob_kwargs))
    support = _support_indices(model, blob, ERROR)
    assert support.size > 0

    dt = model._geometry.dt
    margin = int(
        np.ceil(-blob.width_p * np.log(ERROR * np.sqrt(np.pi)) / abs(blob.v_x * dt))
    )
    support_width = support.max() - support.min() + 1
    assert (stop - start) <= support_width + 2 * margin + 2


@pytest.mark.parametrize(
    "blob_kwargs",
    [
        dict(pos_x0=200.0, v_x=1.0, t_init=0.0),  # far right, moving further right
        dict(pos_x0=-200.0, v_x=-1.0, t_init=0.0),  # far left, moving further left
    ],
)
def test_compute_start_stop_empty_when_blob_never_enters(blob_kwargs):
    """
    A blob that never reaches the domain within the time window contributes
    nothing, so the window must collapse (start == stop) rather than default
    to something non-empty.
    """
    start, stop, model, blob = _start_stop(SingleBlobFactory(**blob_kwargs))
    assert _support_indices(model, blob, ERROR).size == 0  # guard: truly absent
    assert start == stop


def test_compute_start_stop_widens_as_error_decreases():
    """
    A smaller `error` keeps more of the blob's tail, so the window can only
    grow (never shrink) as error decreases -- and must still cover the
    error-dependent support at every step.
    """
    factory = SingleBlobFactory(pos_x0=3.0, v_x=2.0, t_init=0.0)
    widths = []
    for error in [1e-2, 1e-5, 1e-10, 1e-14]:
        start, stop, model, blob = _start_stop(factory, error=error)
        support = _support_indices(model, blob, error)
        assert start <= support.min()
        assert stop > support.max()
        widths.append(stop - start)
    assert widths == sorted(widths)  # non-decreasing as error shrinks
