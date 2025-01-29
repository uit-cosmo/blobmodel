from enum import Enum
from abc import ABC, abstractmethod
import numpy as np


class DistributionEnum(Enum):
    """Enum class used to identify distribution functions."""

    deg = 1
    zeros = 2
    exp = 3
    gamma = 4
    normal = 5
    uniform = 6
    rayleigh = 7


class AbstractDistribution(ABC):
    """Abstract class used to represent and implement a distribution function."""

    @abstractmethod
    def sample(
        self,
        num_blobs: int,
        **kwargs,
    ) -> np.ndarray:
        raise NotImplementedError


def _sample_deg(num_blobs, **kwargs):
    free_param = kwargs["free_param"]
    return free_param * np.ones(num_blobs).astype(np.float64)


def _sample_zeros(num_blobs, **kwargs):
    return np.zeros(num_blobs).astype(np.float64)


def _sample_exp(num_blobs, **kwargs):
    free_param = kwargs["free_param"]
    return np.random.exponential(scale=free_param, size=num_blobs).astype(np.float64)


def _sample_gamma(num_blobs, **kwargs):
    free_param = kwargs["free_param"]
    return np.random.gamma(
        shape=free_param, scale=1 / free_param, size=num_blobs
    ).astype(np.float64)


def _sample_normal(num_blobs, **kwargs):
    free_param = kwargs["free_param"]
    return np.random.normal(loc=0, scale=free_param, size=num_blobs).astype(np.float64)


def _sample_uniform(num_blobs, **kwargs):
    free_param = kwargs["free_param"]
    return np.random.uniform(
        low=1 - free_param / 2, high=1 + free_param / 2, size=num_blobs
    ).astype(np.float64)


def _sample_rayleigh(num_blobs, **kwargs):
    return np.random.rayleigh(scale=np.sqrt(2.0 / np.pi), size=num_blobs).astype(
        np.float64
    )


DISTRIBUTIONS = {
    DistributionEnum.deg: _sample_deg,
    DistributionEnum.zeros: _sample_zeros,
    DistributionEnum.exp: _sample_exp,
    DistributionEnum.gamma: _sample_gamma,
    DistributionEnum.normal: _sample_normal,
    DistributionEnum.uniform: _sample_uniform,
    DistributionEnum.rayleigh: _sample_rayleigh,
}
