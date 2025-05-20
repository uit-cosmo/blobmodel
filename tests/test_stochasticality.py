import pytest
from blobmodel import DefaultBlobFactory, BlobShapeImpl, BlobFactory, DistributionEnum


def test_mean_of_distribution():
    """
    Checks that the distributions are normalized.
    """
    bf = DefaultBlobFactory()
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
    Checks that a KeyError is thrown when using unknown strings as distributions.
    """
    with pytest.raises(AssertionError):
        bf = DefaultBlobFactory(A_dist="something_different")
        bf.sample_blobs(1, 1, 1, BlobShapeImpl(), 1)
