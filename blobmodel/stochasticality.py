from abc import ABC, abstractmethod
import numpy as np
from nptyping import NDArray, Float
from typing import Any, List
from .blobs import Blob
from .pulse_shape import AbstractBlobShape


class BlobFactory(ABC):
    """Abstract class used by 2d propagating blob model to specify blob
    parameters."""

    @abstractmethod
    def sample_blobs(
        self,
        Ly: float,
        T: float,
        num_blobs: int,
        blob_shape: AbstractBlobShape,
        t_drain: float,
    ) -> List[Blob]:
        """creates list of Blobs used in Model."""
        raise NotImplementedError

    @abstractmethod
    def is_one_dimensional(self) -> bool:
        """returns True if the BlobFactory is compatible with a one_dimensional
        model."""
        raise NotImplementedError


class DefaultBlobFactory(BlobFactory):
    """Default implementation of BlobFactory.

    Generates blob parameters for different possible random
    distributions. All random variables are independent from each other
    """

    def __init__(
        self,
        A_dist: str = "exp",
        wx_dist: str = "deg",
        wy_dist: str = "deg",
        vx_dist: str = "deg",
        vy_dist: str = "normal",
        A_parameter: float = 1.0,
        wx_parameter: float = 1.0,
        wy_parameter: float = 1.0,
        vx_parameter: float = 1.0,
        vy_parameter: float = 1.0,
    ) -> None:
        """The following distributions are implemented:

        exp: exponential distribution with mean 1
        gamma: gamma distribution with `free_parameter` as shape parameter and mean 1
        normal: normal distribution with zero mean and `free_parameter` as scale parameter
        uniform: uniorm distribution with mean 1 and `free_parameter` as width
        ray: rayleight distribution with mean 1
        deg: degenerate distribution at `free_parameter`
        zeros: array of zeros
        """
        self.A_dist = A_dist
        self.wx_dist = wx_dist
        self.wy_dist = wy_dist
        self.vx_dist = vx_dist
        self.vy_dist = vy_dist
        self.A_parameter = A_parameter
        self.wx_parameter = wx_parameter
        self.wy_parameter = wy_parameter
        self.vx_parameter = vx_parameter
        self.vy_parameter = vy_parameter

    def _draw_random_variables(
        self,
        dist_type: str,
        free_parameter: float,
        num_blobs: int,
    ) -> NDArray[Any, Float[64]]:

        if dist_type == "exp":
            return np.random.exponential(scale=1, size=num_blobs)
        elif dist_type == "gamma":
            return np.random.gamma(
                shape=free_parameter, scale=1 / free_parameter, size=num_blobs
            )
        elif dist_type == "normal":
            return np.random.normal(loc=0, scale=free_parameter, size=num_blobs)
        elif dist_type == "uniform":
            return np.random.uniform(
                low=1 - free_parameter / 2, high=1 + free_parameter / 2, size=num_blobs
            )
        elif dist_type == "ray":
            return np.random.rayleigh(scale=np.sqrt(2.0 / np.pi), size=num_blobs)
        elif dist_type == "deg":
            return free_parameter * np.ones(num_blobs)
        elif dist_type == "zeros":
            return np.zeros(num_blobs)
        else:
            raise NotImplementedError(
                self.__class__.__name__ + ".distribution function not implemented"
            )

    def sample_blobs(
        self,
        Ly: float,
        T: float,
        num_blobs: int,
        blob_shape: AbstractBlobShape,
        t_drain: float,
    ) -> List[Blob]:
        amps = self._draw_random_variables(
            dist_type=self.A_dist, free_parameter=self.A_parameter, num_blobs=num_blobs
        )
        wxs = self._draw_random_variables(self.wx_dist, self.wx_parameter, num_blobs)
        wys = self._draw_random_variables(self.wy_dist, self.wy_parameter, num_blobs)
        vxs = self._draw_random_variables(self.vx_dist, self.vx_parameter, num_blobs)
        vys = self._draw_random_variables(self.vy_dist, self.vy_parameter, num_blobs)
        posxs = np.zeros(num_blobs)
        posys = np.random.uniform(low=0.0, high=Ly, size=num_blobs)
        t_inits = np.random.uniform(low=0, high=T, size=num_blobs)

        blobs = [
            Blob(
                blob_id=i,
                blob_shape=blob_shape,
                amplitude=amps[i],
                width_prop=wxs[i],
                width_perp=wys[i],
                v_x=vxs[i],
                v_y=vys[i],
                pos_x=posxs[i],
                pos_y=posys[i],
                t_init=t_inits[i],
                t_drain=t_drain,
            )
            for i in range(num_blobs)
        ]

        # sort blobs by amplitude
        return np.array(blobs)[np.argsort(amps)]

    def is_one_dimensional(self) -> bool:
        # Perpendicular width parameters are irrelevant since perp shape should be ignored by the Bolb class.
        return self.vy_dist == "zeros"
