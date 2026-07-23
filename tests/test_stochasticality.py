import numpy as np
import pytest
from blobmodel import BlobShapeImpl, DefaultBlobFactory, DistributionEnum


def test_mean_of_distribution():
    """
    Checks that the distributions are normalized.
    """
    bf = DefaultBlobFactory(seed=42)
    distributions_mean_1 = [
        DistributionEnum.exp,
        DistributionEnum.gamma,
        DistributionEnum.uniform,
        DistributionEnum.rayleigh,
        DistributionEnum.deg,
    ]
    distributions_mean_0 = [DistributionEnum.normal, DistributionEnum.zeros]

    for dist in distributions_mean_1:
        bf.set_sampler("amplitude", dist)
        tmp = bf._draw_random_variables("amplitude", 10000)
        assert 0.95 <= tmp.mean() <= 1.05

    for dist in distributions_mean_0:
        bf.set_sampler("amplitude", dist)
        tmp = bf._draw_random_variables("amplitude", 10000)
        assert -0.05 <= tmp.mean() <= 0.05


def test_not_implemented_distribution():
    """
    Checks that a TypeError is thrown when using unknown strings as samplers.
    """
    with pytest.raises(TypeError, match="amplitude"):
        DefaultBlobFactory().set_sampler("amplitude", "something_different")


def test_unknown_parameter_raises():
    """set_sampler must reject parameter names it does not know."""
    with pytest.raises(ValueError, match="Unknown parameter"):
        DefaultBlobFactory().set_sampler("A", DistributionEnum.exp)


def test_uniform_width_parameter_too_big_raises():
    """
    A uniform width distribution with free_parameter > 2 would produce negative
    widths and must be rejected.
    """
    with pytest.raises(ValueError, match="wp"):
        DefaultBlobFactory().set_sampler("wp", DistributionEnum.uniform, 3)
    with pytest.raises(ValueError, match="ws"):
        DefaultBlobFactory().set_sampler("ws", DistributionEnum.uniform, 2.5)


def test_uniform_width_parameter_at_limit_accepted():
    """
    free_parameter = 2 (support [0, 2]) and non-uniform distributions with big
    parameters are still accepted.
    """
    DefaultBlobFactory().set_sampler("wp", DistributionEnum.uniform, 2)
    DefaultBlobFactory().set_sampler("wp", DistributionEnum.exp, 3)


def test_rayleigh_ignores_free_parameter():
    """
    The rayleigh distribution intentionally ignores its free parameter: the
    scale is fixed to sqrt(2 / pi) so the mean is always 1.
    """
    bf = DefaultBlobFactory(seed=42)
    for free_parameter in [1, 100]:
        bf.set_sampler("amplitude", DistributionEnum.rayleigh, free_parameter)
        tmp = bf._draw_random_variables("amplitude", 10000)
        assert 0.95 <= tmp.mean() <= 1.05


def test_callable_sampler_used_for_blobs():
    """A callable sampler's values must end up on the sampled blobs."""
    bf = DefaultBlobFactory().set_sampler(
        "amplitude", lambda rng, num_blobs: np.arange(1.0, num_blobs + 1)
    )
    blobs = bf.sample_blobs(Ly=10, T=10, num_blobs=3, blob_shape=BlobShapeImpl())
    assert sorted(b.amplitude for b in blobs) == [1.0, 2.0, 3.0]


def test_callable_sampler_receives_factory_rng():
    """
    The callable is given the factory's generator, so drawing from it makes
    two same-seeded factories produce identical blobs.
    """

    def sampler(rng, num_blobs):
        return rng.exponential(size=num_blobs)

    blobs = [
        DefaultBlobFactory(seed=42)
        .set_sampler("amplitude", sampler)
        .sample_blobs(Ly=10, T=10, num_blobs=5, blob_shape=BlobShapeImpl())
        for _ in range(2)
    ]
    for b1, b2 in zip(*blobs):
        assert b1.amplitude == b2.amplitude


def test_callable_sampler_with_free_parameter_raises():
    """free_parameter has no meaning for a callable sampler."""
    with pytest.raises(ValueError, match="free_parameter"):
        DefaultBlobFactory().set_sampler(
            "amplitude", lambda rng, num_blobs: np.ones(num_blobs), 2.0
        )


def test_callable_sampler_wrong_shape_raises():
    """A sampler returning the wrong number of values must be rejected."""
    bf = DefaultBlobFactory().set_sampler(
        "amplitude", lambda rng, num_blobs: np.ones(num_blobs + 1)
    )
    with pytest.raises(ValueError, match="amplitude"):
        bf.sample_blobs(Ly=10, T=10, num_blobs=3, blob_shape=BlobShapeImpl())


def test_default_factory_defaults():
    """
    A bare DefaultBlobFactory samples exponential amplitudes with mean 1 (the
    canonical FPP choice), zero perpendicular velocity (matching the Blob
    default v_y=0) and constant values for everything else: widths and vx 1,
    shape parameters 0.5.
    """
    bf = DefaultBlobFactory(seed=42)
    blobs = bf.sample_blobs(Ly=10, T=10, num_blobs=1000, blob_shape=BlobShapeImpl())
    amps = np.array([b.amplitude for b in blobs])
    assert 0.9 <= amps.mean() <= 1.1
    assert amps.std() > 0
    assert all(b.width_p == 1 and b.width_s == 1 for b in blobs)
    assert all(b.v_x == 1 and b.v_y == 0 for b in blobs)


def test_is_one_dimensional_only_for_zeros_enum():
    """
    Only the built-in zeros distribution for "vy" marks the factory as
    one-dimensional (the default since vy defaults to zeros); a callable
    sampler does not, even if it returns zeros.
    """
    assert DefaultBlobFactory().is_one_dimensional()
    assert not (
        DefaultBlobFactory()
        .set_sampler("vy", DistributionEnum.deg)
        .is_one_dimensional()
    )
    assert not (
        DefaultBlobFactory()
        .set_sampler("vy", lambda rng, num_blobs: np.zeros(num_blobs))
        .is_one_dimensional()
    )
