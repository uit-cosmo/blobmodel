"""This module defines a class for generating blob parameters."""

from abc import ABC, abstractmethod
import numpy as np
from nptyping import NDArray, Float
from typing import Any, List, Union, Dict
from .blobs import Blob
from .blob_shape import AbstractBlobShape


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
        t_drain: Union[float, NDArray],
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
        spx_dist: str = "deg",
        spy_dist: str = "deg",
        A_parameter: float = 1.0,
        wx_parameter: float = 1.0,
        wy_parameter: float = 1.0,
        vx_parameter: float = 1.0,
        vy_parameter: float = 1.0,
        shape_param_x_parameter: float = 0.5,
        shape_param_y_parameter: float = 0.5,
        blob_alignment: bool = True,
    ) -> None:
        """
        Default implementation of BlobFactory.

        Generates blob parameters for different possible random distributions.
        All random variables are independent from each other.

        Parameters
        ----------
        A_dist : str, optional
            Distribution type for amplitude, by default "exp"
        wx_dist : str, optional
            Distribution type for width in the x-direction, by default "deg"
        wy_dist : str, optional
            Distribution type for width in the y-direction, by default "deg"
        vx_dist : str, optional
            Distribution type for velocity in the x-direction, by default "deg"
        vy_dist : str, optional
            Distribution type for velocity in the y-direction, by default "normal"
        spx_dist : str, optional
            Distribution type for shape parameter in the x-direction, by default "deg"
        spy_dist : str, optional
            Distribution type for shape parameter in the y-direction, by default "deg"
        A_parameter : float, optional
            Free parameter for the amplitude distribution, by default 1.0
        wx_parameter : float, optional
            Free parameter for the width distribution in the x-direction, by default 1.0
        wy_parameter : float, optional
            Free parameter for the width distribution in the y-direction, by default 1.0
        vx_parameter : float, optional
            Free parameter for the velocity distribution in the x-direction, by default 1.0
        vy_parameter : float, optional
            Free parameter for the velocity distribution in the y-direction, by default 1.0
        shape_param_x_parameter : float, optional
            Free parameter for the shape parameter distribution in the x-direction, by default 0.5
        shape_param_y_parameter : float, optional
            Free parameter for the shape parameter distribution in the y-direction, by default 0.5
        blob_alignment : bool, optional
            If blob_aligment == True, the blob shapes are rotated in the propagation direction of the blob
            If blob_aligment == False, the blob shapes are independent of the propagation direction

        Notes
        -----
        - The following distributions are implemented:
            - exp: exponential distribution with mean 1
            - gamma: gamma distribution with `free_parameter` as shape parameter and mean 1
            - normal: normal distribution with zero mean and `free_parameter` as scale parameter
            - uniform: uniorm distribution with mean 1 and `free_parameter` as width
            - ray: rayleight distribution with mean 1
            - deg: degenerate distribution at `free_parameter`
            - zeros: array of zeros

        """
        self.amplitude_dist = A_dist
        self.width_x_dist = wx_dist
        self.width_y_dist = wy_dist
        self.velocity_x_dist = vx_dist
        self.velocity_y_dist = vy_dist
        self.shape_param_x_dist = spx_dist
        self.shape_param_y_dist = spy_dist
        self.amplitude_parameter = A_parameter
        self.width_x_parameter = wx_parameter
        self.width_y_parameter = wy_parameter
        self.velocity_x_parameter = vx_parameter
        self.velocity_y_parameter = vy_parameter
        self.shape_param_x_parameter = shape_param_x_parameter
        self.shape_param_y_parameter = shape_param_y_parameter
        self.blob_alignment = blob_alignment

    def _draw_random_variables(
        self,
        dist_type: str,
        free_parameter: float,
        num_blobs: int,
    ) -> np.ndarray:
        """
        Draws random variables from a specified distribution.

        Parameters
        ----------
        dist_type : str
            Type of distribution.
        free_parameter : float
            Free parameter for the distribution.
        num_blobs : int
            Number of random variables to draw.

        Returns
        -------
        NDArray[Any, Float[64]]
            Array of random variables drawn from the specified distribution.
        """
        if dist_type == "exp":
            return np.random.exponential(scale=1, size=num_blobs).astype(np.float64)
        elif dist_type == "gamma":
            return np.random.gamma(
                shape=free_parameter, scale=1 / free_parameter, size=num_blobs
            ).astype(np.float64)
        elif dist_type == "normal":
            return np.random.normal(loc=0, scale=free_parameter, size=num_blobs).astype(
                np.float64
            )
        elif dist_type == "uniform":
            return np.random.uniform(
                low=1 - free_parameter / 2, high=1 + free_parameter / 2, size=num_blobs
            ).astype(np.float64)
        elif dist_type == "ray":
            return np.random.rayleigh(
                scale=np.sqrt(2.0 / np.pi), size=num_blobs
            ).astype(np.float64)
        elif dist_type == "deg":
            return free_parameter * np.ones(num_blobs).astype(np.float64)
        elif dist_type == "zeros":
            return np.zeros(num_blobs).astype(np.float64)
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
        t_drain: Union[float, NDArray],
    ) -> List[Blob]:
        """
        Creates a list of Blobs used in the Model.

        Parameters
        ----------
        Ly : float
            Size of the domain in the y-direction.
        T : float
            Total time duration.
        num_blobs : int
            Number of blobs to generate.
        blob_shape : AbstractBlobShape
            Object representing the shape of the blobs.
        t_drain : float
            Time at which the blobs start draining.

        Returns
        -------
        List[Blob]
            List of Blob objects generated for the Model.
        """
        amps = self._draw_random_variables(
            dist_type=self.amplitude_dist,
            free_parameter=self.amplitude_parameter,
            num_blobs=num_blobs,
        )
        wxs = self._draw_random_variables(
            self.width_x_dist, self.width_x_parameter, num_blobs
        )
        wys = self._draw_random_variables(
            self.width_y_dist, self.width_y_parameter, num_blobs
        )
        vxs = self._draw_random_variables(
            self.velocity_x_dist, self.velocity_x_parameter, num_blobs
        )
        vys = self._draw_random_variables(
            self.velocity_y_dist, self.velocity_y_parameter, num_blobs
        )
        spxs = self._draw_random_variables(
            self.shape_param_x_dist, self.shape_param_x_parameter, num_blobs
        )
        spys = self._draw_random_variables(
            self.shape_param_y_dist, self.shape_param_y_parameter, num_blobs
        )
        # For now, only a lambda parameter is implemented
        spxs_dict = [{"lam": s} for s in spxs]
        spys_dict = [{"lam": s} for s in spys]
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
                prop_shape_parameters=spxs_dict[i],
                perp_shape_parameters=spys_dict[i],
                blob_alignment=self.blob_alignment,
            )
            for i in range(num_blobs)
        ]

        # sort blobs by amplitude
        return sorted(blobs, key=lambda x: x.amplitude)

    def is_one_dimensional(self) -> bool:
        """
        Returns True if the BlobFactory is compatible with a one-dimensional model.

        Returns
        -------
        bool
            True if the BlobFactory is compatible with a one-dimensional model,
            False otherwise.

        Notes
        -----
        - Perpendicular width parameters are irrelevant since perp shape should be ignored by the Bolb class.

        """
        return self.velocity_y_dist == "zeros"
