"""Tests for BlobListFactory, CallableBlobFactory and Model.from_blobs."""

import numpy as np
import warnings
from blobmodel import (
    Blob,
    BlobListFactory,
    BlobShapeImpl,
    CallableBlobFactory,
    Geometry,
    Model,
)


def _make_blob(blob_id=0, v_y=0.0, t_drain=np.inf, amplitude=1.0):
    return Blob(
        blob_id=blob_id,
        blob_shape=BlobShapeImpl(),
        amplitude=amplitude,
        width_p=1,
        width_s=1,
        v_x=1,
        v_y=v_y,
        pos_x0=0,
        pos_y0=1,
        t_init=0,
        t_drain=t_drain,
    )


def _geometry():
    return Geometry(Nx=8, Ny=8, Lx=2, Ly=2, dt=0.5, T=2)


def test_blob_list_factory_returns_given_blobs():
    """sample_blobs returns exactly the stored blobs; all arguments are ignored."""
    blobs = [_make_blob(blob_id=i) for i in range(3)]
    bf = BlobListFactory(blobs)
    sampled = bf.sample_blobs(
        Ly=10, T=10, num_blobs=1000, blob_shape=BlobShapeImpl(), t_drain=10
    )
    assert sampled == blobs


def test_blob_list_factory_one_dimensional_inferred():
    """is_one_dimensional is True exactly when every blob has v_y == 0."""
    assert BlobListFactory([_make_blob(v_y=0), _make_blob(v_y=0)]).is_one_dimensional()
    assert not BlobListFactory([_make_blob(v_y=1)]).is_one_dimensional()


def test_one_dimensional_model_from_blobs_no_warning():
    """A 1D model built from v_y == 0 blobs does not emit the 1D-factory warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Model.from_blobs(
            [_make_blob()],
            geometry=Geometry(Nx=8, Ny=1, Lx=2, Ly=0, dt=0.5, T=2),
            one_dimensional=True,
            verbose=False,
        )


def test_from_blobs_matches_explicit_model():
    """Model.from_blobs produces the same realization as the verbose
    trivial-factory + dummy-parameters construction it replaces."""
    blobs = [_make_blob(blob_id=i, amplitude=a) for i, a in enumerate((1.0, 2.0))]
    ds_shortcut = Model.from_blobs(
        blobs, geometry=_geometry(), verbose=False
    ).make_realization()
    ds_explicit = Model(
        geometry=_geometry(),
        num_blobs=len(blobs),
        blob_shape=BlobShapeImpl(),
        t_drain=1e10,
        blob_factory=BlobListFactory(blobs),
        verbose=False,
    ).make_realization()
    np.testing.assert_array_equal(ds_shortcut.n.values, ds_explicit.n.values)


def test_from_blobs_realization_is_repeatable():
    """Realizing the same blob list twice gives identical density fields."""
    blobs = [_make_blob()]
    model = Model.from_blobs(blobs, geometry=_geometry(), verbose=False)
    np.testing.assert_array_equal(
        model.make_realization().n.values, model.make_realization().n.values
    )


def _random_blob_getter(rng):
    return _make_blob(amplitude=rng.exponential(), v_y=0.0)


def test_callable_factory_uses_num_blobs():
    """The getter is called once per requested blob."""
    bf = CallableBlobFactory(_random_blob_getter, seed=42)
    sampled = bf.sample_blobs(
        Ly=10, T=10, num_blobs=7, blob_shape=BlobShapeImpl(), t_drain=10
    )
    assert len(sampled) == 7


def test_callable_factory_same_seed_same_blobs():
    """Two factories with the same seed produce identical blobs."""
    blobs_1 = CallableBlobFactory(_random_blob_getter, seed=42).sample_blobs(
        Ly=10, T=10, num_blobs=10, blob_shape=BlobShapeImpl(), t_drain=10
    )
    blobs_2 = CallableBlobFactory(_random_blob_getter, seed=42).sample_blobs(
        Ly=10, T=10, num_blobs=10, blob_shape=BlobShapeImpl(), t_drain=10
    )
    assert [b.amplitude for b in blobs_1] == [b.amplitude for b in blobs_2]


def test_callable_factory_model_seed_overrides():
    """A seed passed to Model replaces the factory's generator, making
    realizations reproducible regardless of the factory's own seed."""

    def _model(factory_seed):
        return Model(
            geometry=_geometry(),
            num_blobs=10,
            blob_factory=CallableBlobFactory(_random_blob_getter, seed=factory_seed),
            verbose=False,
            seed=42,
        )

    np.testing.assert_array_equal(
        _model(factory_seed=1).make_realization().n.values,
        _model(factory_seed=2).make_realization().n.values,
    )


def test_callable_factory_one_dimensional_flag():
    """is_one_dimensional reflects the declared flag."""
    assert not CallableBlobFactory(_random_blob_getter).is_one_dimensional()
    assert CallableBlobFactory(
        _random_blob_getter, one_dimensional=True
    ).is_one_dimensional()


def test_t_drain_inf_means_no_draining():
    """t_drain=np.inf disables the exponential decay: a blob passing through
    the domain keeps its full amplitude."""
    geometry = Geometry(Nx=10, Ny=1, Lx=10, Ly=0, dt=1, T=10, periodic_y=False)
    blob_inf = _make_blob(t_drain=np.inf)
    ds = Model.from_blobs(
        [blob_inf], geometry=geometry, one_dimensional=True, verbose=False
    ).make_realization()
    # The blob center sits at x = t; the peak amplitude must not decay.
    peaks = ds.n.values[0, np.arange(10), np.arange(10)]
    np.testing.assert_allclose(peaks, peaks[0])

    # And it must exceed the equivalent draining blob at late times.
    blob_drain = _make_blob(t_drain=1)
    ds_drain = Model.from_blobs(
        [blob_drain], geometry=geometry, one_dimensional=True, verbose=False
    ).make_realization()
    assert ds_drain.n.values[0, -1, -1] < ds.n.values[0, -1, -1]


def test_model_accepts_t_drain_inf():
    """Model validation accepts t_drain=np.inf (documented no-drain value)."""
    Model(geometry=_geometry(), t_drain=np.inf, verbose=False)
