"""Tests for the seeding mechanism (reproducible realizations)."""

import numpy as np
import pytest
from blobmodel import (
    Geometry,
    Model,
    DefaultBlobFactory,
    DistributionEnum,
    BlobShapeImpl,
)


def _blob_attributes(blob):
    return (
        blob.amplitude,
        blob.width_p,
        blob.width_s,
        blob.v_x,
        blob.v_y,
        blob.pos_x0,
        blob.pos_y0,
        blob.t_init,
    )


def _sample_default_blobs(factory, num_blobs=10):
    return factory.sample_blobs(
        Ly=10, T=10, num_blobs=num_blobs, blob_shape=BlobShapeImpl(), t_drain=10
    )


def _random_model(seed=None, blob_factory=None):
    return Model(
        geometry=Geometry(Nx=8, Ny=8, Lx=2, Ly=2, dt=0.5, T=2),
        num_blobs=10,
        blob_factory=blob_factory,
        verbose=False,
        seed=seed,
    )


def test_factory_same_seed_same_blobs():
    """Two factories constructed with the same seed produce identical blobs."""
    bf_1 = DefaultBlobFactory(A_dist=DistributionEnum.exp, seed=42)
    bf_2 = DefaultBlobFactory(A_dist=DistributionEnum.exp, seed=42)
    blobs_1 = _sample_default_blobs(bf_1)
    blobs_2 = _sample_default_blobs(bf_2)
    for b1, b2 in zip(blobs_1, blobs_2):
        assert _blob_attributes(b1) == _blob_attributes(b2)


def test_factory_different_seed_different_blobs():
    """Different seeds produce different blobs."""
    bf_1 = DefaultBlobFactory(seed=42)
    bf_2 = DefaultBlobFactory(seed=43)
    blobs_1 = _sample_default_blobs(bf_1)
    blobs_2 = _sample_default_blobs(bf_2)
    assert any(
        _blob_attributes(b1) != _blob_attributes(b2) for b1, b2 in zip(blobs_1, blobs_2)
    )


def test_factory_accepts_generator():
    """A numpy Generator can be passed instead of an integer seed."""
    bf_1 = DefaultBlobFactory(seed=np.random.default_rng(42))
    bf_2 = DefaultBlobFactory(seed=42)
    blobs_1 = _sample_default_blobs(bf_1)
    blobs_2 = _sample_default_blobs(bf_2)
    for b1, b2 in zip(blobs_1, blobs_2):
        assert _blob_attributes(b1) == _blob_attributes(b2)


@pytest.mark.parametrize("dist", list(DistributionEnum))
def test_all_distributions_reproducible(dist):
    """The rng is threaded through every distribution sampling function."""
    bf_1 = DefaultBlobFactory(seed=7)
    bf_2 = DefaultBlobFactory(seed=7)
    draws_1 = bf_1._draw_random_variables(dist, free_parameter=1, num_blobs=100)
    draws_2 = bf_2._draw_random_variables(dist, free_parameter=1, num_blobs=100)
    np.testing.assert_array_equal(draws_1, draws_2)


def test_model_same_seed_same_realization():
    """Two models with the same seed produce identical density fields."""
    ds_1 = _random_model(seed=42).make_realization()
    ds_2 = _random_model(seed=42).make_realization()
    np.testing.assert_array_equal(ds_1.n.values, ds_2.n.values)


def test_model_different_seed_different_realization():
    """Different seeds produce different density fields."""
    ds_1 = _random_model(seed=42).make_realization()
    ds_2 = _random_model(seed=43).make_realization()
    assert not np.array_equal(ds_1.n.values, ds_2.n.values)


def test_model_seed_overrides_factory_seed():
    """A seed passed to Model replaces the factory's own generator."""
    ds_1 = _random_model(
        seed=42, blob_factory=DefaultBlobFactory(seed=1)
    ).make_realization()
    ds_2 = _random_model(
        seed=42, blob_factory=DefaultBlobFactory(seed=2)
    ).make_realization()
    np.testing.assert_array_equal(ds_1.n.values, ds_2.n.values)


def test_default_factories_not_shared():
    """Each default-constructed Model gets its own factory instance, so seeding
    one model cannot affect another."""
    model_1 = _random_model(seed=42)
    model_2 = _random_model(seed=43)
    assert model_1._blob_factory is not model_2._blob_factory
    # Realizations remain reproducible regardless of construction order.
    ds_2 = model_2.make_realization()
    ds_1 = model_1.make_realization()
    np.testing.assert_array_equal(
        ds_1.n.values, _random_model(seed=42).make_realization().n.values
    )
    np.testing.assert_array_equal(
        ds_2.n.values, _random_model(seed=43).make_realization().n.values
    )
