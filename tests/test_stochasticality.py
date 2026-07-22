import pytest
from blobmodel import DefaultBlobFactory, BlobShapeImpl, BlobFactory, DistributionEnum


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
        tmp = bf._draw_random_variables(
            dist=dist,
            free_parameter=1,
            num_blobs=10000,
        )
        assert 0.95 <= tmp.mean() <= 1.05

    for dist in distributions_mean_0:
        tmp = bf._draw_random_variables(
            dist=dist,
            free_parameter=1,
            num_blobs=10000,
        )
        assert -0.05 <= tmp.mean() <= 0.05


def test_not_implemented_distribution():
    """
    Checks that a TypeError is thrown when using unknown strings as distributions.
    """
    with pytest.raises(TypeError, match="A_dist"):
        DefaultBlobFactory(A_dist="something_different")


def test_uniform_width_parameter_too_big_raises():
    """
    A uniform width distribution with free_parameter > 2 would produce negative
    widths and must be rejected.
    """
    with pytest.raises(ValueError, match="wp_parameter"):
        DefaultBlobFactory(wp_dist=DistributionEnum.uniform, wp_parameter=3)
    with pytest.raises(ValueError, match="ws_parameter"):
        DefaultBlobFactory(ws_dist=DistributionEnum.uniform, ws_parameter=2.5)


def test_uniform_width_parameter_at_limit_accepted():
    """
    free_parameter = 2 (support [0, 2]) and non-uniform distributions with big
    parameters are still accepted.
    """
    DefaultBlobFactory(wp_dist=DistributionEnum.uniform, wp_parameter=2)
    DefaultBlobFactory(wp_dist=DistributionEnum.exp, wp_parameter=3)


def test_rayleigh_ignores_free_parameter():
    """
    The rayleigh distribution intentionally ignores its free parameter: the
    scale is fixed to sqrt(2 / pi) so the mean is always 1.
    """
    bf = DefaultBlobFactory(seed=42)
    for free_parameter in [1, 100]:
        tmp = bf._draw_random_variables(
            dist=DistributionEnum.rayleigh,
            free_parameter=free_parameter,
            num_blobs=10000,
        )
        assert 0.95 <= tmp.mean() <= 1.05
