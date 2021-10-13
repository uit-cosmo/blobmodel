from abc import ABC, abstractmethod
import numpy as np
from .blobs import Blob
from nptyping import NDArray, Float
from typing import Any


class BlobFactory(ABC):
    """
    Abstract class used by 2d propagating blob model to specify blob parameters. 
    """

    @abstractmethod
    def sample_blobs(
        self, Ly: float, T: float, num_blobs: int, blob_shape: str, t_drain: float
    ) -> list[Blob]:
        raise NotImplementedError


class DefaultBlobFactory(BlobFactory):
    """
    Defualt blob parameters 
    """

    def __init__(
        self,
        A_dist: str = "exp",
        W_dist: str = "exp",
        vx_dist: str = "deg",
        vy_dist: str = "normal",
        A_parameter: float = 1.0,
        W_parameter: float = 1.0,
        vx_parameter: float = 1.0,
        vy_parameter: float = 1.0,
    ) -> None:
        self.A_dist = A_dist
        self.W_dist = W_dist
        self.vx_dist = vx_dist
        self.vy_dist = vy_dist
        self.A_parameter = A_parameter
        self.W_parameter = W_parameter
        self.vx_parameter = vx_parameter
        self.vy_parameter = vy_parameter

    def __choose_distribution(
        self, dist_type: str, free_parameter: float, num_blobs: int,
    ) -> NDArray[Any, Float[64]]:
        """
        The following distributions are implemented:
            exp: exponential distribution with mean 1
            gamma: gamma distribution with `free_parameter` as shape parameter and mean 1
            normal: normal distribution with zero mean and `free_parameter` as scale parameter
            uniform: uniorm distribution with mean 1 and `free_parameter` as width
            ray: rayleight distribution with mean 1
            deg: array on ones
            zeros: array of zeros
        """
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
            return np.ones(num_blobs)
        elif dist_type == "zeros":
            return np.zeros(num_blobs)
        else:
            raise NotImplementedError(
                self.__class__.__name__ + ".distribution function not implemented"
            )

    def sample_blobs(
        self, Ly: float, T: float, num_blobs: int, blob_shape: str, t_drain: float
    ) -> list[Blob]:
        __amp = self.__choose_distribution(
            dist_type=self.A_dist, free_parameter=self.A_parameter, num_blobs=num_blobs
        )
        __width = self.__choose_distribution(self.W_dist, self.W_parameter, num_blobs)
        __vx = self.__choose_distribution(self.vx_dist, self.vx_parameter, num_blobs)
        __vy = self.__choose_distribution(self.vy_dist, self.vy_parameter, num_blobs)
        __posx = np.zeros(num_blobs)
        __posy = np.random.uniform(low=0.0, high=Ly, size=num_blobs)
        __t_init = np.random.uniform(low=0, high=T, size=num_blobs)

        # sort blobs by __t_init
        __t_init = np.sort(__t_init)

        __blobs = []
        for i in range(num_blobs):
            __blobs.append(
                Blob(
                    id=i,
                    blob_shape=blob_shape,
                    amplitude=__amp[i],
                    width_x=__width[i],
                    width_y=__width[i],
                    v_x=__vx[i],
                    v_y=__vy[i],
                    pos_x=__posx[i],
                    pos_y=__posy[i],
                    t_init=__t_init[i],
                    t_drain=t_drain,
                )
            )
        return __blobs
