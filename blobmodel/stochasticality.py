"""This module defines a class for generating blob parameters."""

from abc import ABC, abstractmethod
from nptyping import NDArray
from typing import List, Union, Callable
import numpy as np
from .blobs import Blob
from .blob_shape import AbstractBlobShape
from .distributions import DISTRIBUTIONS, DistributionEnum


class BlobFactory(ABC):
    """
    Abstract class used by 2d propagating blob model to specify blob
    parameters.
    """

    @abstractmethod
    def sample_blobs(
        self,
        Ly: float,
        T: float,
        num_blobs: int,
        blob_shape: AbstractBlobShape,
        t_drain: Union[float, NDArray],
    ) -> List[Blob]:
        """
        Creates a list of Blobs used in Model.
        """
        raise NotImplementedError

    @abstractmethod
    def is_one_dimensional(self) -> bool:
        """returns True if the BlobFactory is compatible with a one_dimensional
        model."""
        raise NotImplementedError

    def set_rng(self, rng: np.random.Generator) -> None:
        """
        Set the random number generator used when sampling blob parameters.

        Called by `Model` when it is given a `seed`. The default implementation
        stores the generator as `self.rng`; custom factories that want to be
        seedable through `Model` should draw their random numbers from
        `self.rng`.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator to use for sampling.
        """
        self.rng = rng


class DefaultBlobFactory(BlobFactory):
    """Default implementation of BlobFactory.

    Generates blob parameters for different possible random
    distributions. All random variables are independent from each other
    """

    def __init__(
        self,
        A_dist: DistributionEnum = DistributionEnum.exp,
        wp_dist: DistributionEnum = DistributionEnum.deg,
        ws_dist: DistributionEnum = DistributionEnum.deg,
        vx_dist: DistributionEnum = DistributionEnum.deg,
        vy_dist: DistributionEnum = DistributionEnum.deg,
        spp_dist: DistributionEnum = DistributionEnum.deg,
        sps_dist: DistributionEnum = DistributionEnum.deg,
        A_parameter: float = 1.0,
        wp_parameter: float = 1.0,
        ws_parameter: float = 1.0,
        vx_parameter: float = 1.0,
        vy_parameter: float = 1.0,
        shape_param_p_parameter: float = 0.5,
        shape_param_s_parameter: float = 0.5,
        blob_alignment: bool = False,
        seed: Union[int, np.random.Generator, None] = None,
    ) -> None:
        """
        Default implementation of BlobFactory.

        Generates blob parameters for different possible random distributions.
        All random variables are independent from each other.

        Parameters
        ----------
        A_dist : Distribution, optional
            Distribution type for amplitude, by default "Distribution.exp"
        wp_dist : Distribution, optional
            Distribution type for width in the principal blob direction, by default "Distribution.deg"
        ws_dist : Distribution, optional
            Distribution type for width in the secondary blob direction, by default "Distribution.deg"
        vx_dist : Distribution, optional
            Distribution type for velocity in the x-direction, by default "Distribution.deg"
        vy_dist : Distribution, optional
            Distribution type for velocity in the y-direction, by default "Distribution.deg"
        spp_dist : Distribution, optional
            Distribution type for shape parameter in the principal blob direction, by default "Distribution.deg"
        sps_dist : Distribution, optional
            Distribution type for shape parameter in the secondary blob direction, by default "Distribution.deg"
        A_parameter : float, optional
            Free parameter for the amplitude distribution, by default 1.0
        wp_parameter : float, optional
            Free parameter for the width distribution in the principal direction, by default 1.0
        ws_parameter : float, optional
            Free parameter for the width distribution in the secondary direction, by default 1.0
        vx_parameter : float, optional
            Free parameter for the velocity distribution in the x-direction, by default 1.0
        vy_parameter : float, optional
            Free parameter for the velocity distribution in the y-direction, by default 1.0
        shape_param_p_parameter : float, optional
            Free parameter for the shape parameter distribution in the principal direction, by default 0.5
        shape_param_s_parameter : float, optional
            Free parameter for the shape parameter distribution in the secondary direction, by default 0.5
        blob_alignment : bool, optional
            If blob_alignment == True, the blob shapes are rotated in the propagation direction of the blob.
            If blob_alignment == False, the blob shapes are independent of the propagation direction.
            By default False (matching the Blob class default). This is ignored
            once a tilt angle has been registered with `set_theta_setter`.
        seed : int, np.random.Generator or None, optional
            Seed (or an already constructed `numpy.random.Generator`) for the
            random number generator used to sample blob parameters. Two
            factories constructed with the same seed produce identical blobs.
            By default None, i.e. a freshly seeded generator (non-reproducible).
            Note that a seed passed to `Model` takes precedence: it replaces
            this factory's generator via `set_rng`.

        Notes
        -----
        - The following distributions are implemented:
            - exp: exponential distribution with mean 1
            - gamma: gamma distribution with `free_parameter` as shape parameter and mean 1
            - normal: normal distribution with zero mean and `free_parameter` as scale parameter
            - uniform: uniform distribution with mean 1 and `free_parameter` as width,
              i.e. support [1 - `free_parameter` / 2, 1 + `free_parameter` / 2].
              `free_parameter` > 2 produces negative samples and is therefore
              rejected with a ValueError for the width distributions.
            - rayleigh: rayleigh distribution with mean 1. `free_parameter` is
              intentionally ignored: the scale is fixed to sqrt(2 / pi).
            - deg: degenerate distribution at `free_parameter`
            - zeros: array of zeros

        Raises
        ------
        TypeError
            If a distribution argument is not a DistributionEnum member.
        ValueError
            If a width distribution (`wp_dist` or `ws_dist`) is uniform with
            `free_parameter` > 2, which would produce negative blob widths.

        """
        for name, dist in (
            ("A_dist", A_dist),
            ("wp_dist", wp_dist),
            ("ws_dist", ws_dist),
            ("vx_dist", vx_dist),
            ("vy_dist", vy_dist),
            ("spp_dist", spp_dist),
            ("sps_dist", sps_dist),
        ):
            if not isinstance(dist, DistributionEnum):
                raise TypeError(
                    f"{name} must be a DistributionEnum, got {type(dist).__name__}."
                )

        for name, dist, parameter in (
            ("wp", wp_dist, wp_parameter),
            ("ws", ws_dist, ws_parameter),
        ):
            if dist == DistributionEnum.uniform and parameter > 2:
                raise ValueError(
                    f"{name}_parameter = {parameter} with {name}_dist = uniform would produce "
                    f"negative blob widths: the uniform distribution has support "
                    f"[1 - {name}_parameter / 2, 1 + {name}_parameter / 2], so {name}_parameter must be <= 2."
                )

        self.amplitude_dist = A_dist
        self.width_p_dist = wp_dist
        self.width_s_dist = ws_dist
        self.velocity_x_dist = vx_dist
        self.velocity_y_dist = vy_dist
        self.shape_param_p_dist = spp_dist
        self.shape_param_s_dist = sps_dist
        self.amplitude_parameter = A_parameter
        self.width_p_parameter = wp_parameter
        self.width_s_parameter = ws_parameter
        self.velocity_x_parameter = vx_parameter
        self.velocity_y_parameter = vy_parameter
        self.shape_param_p_parameter = shape_param_p_parameter
        self.shape_param_s_parameter = shape_param_s_parameter
        self.blob_alignment = blob_alignment
        self.theta_setter: Union[Callable[[], float], None] = None
        self.rng = np.random.default_rng(seed)

    def _draw_random_variables(
        self, dist: DistributionEnum, free_parameter: float, num_blobs: int
    ) -> np.ndarray:
        """Draws random variables from a specified distribution."""
        return DISTRIBUTIONS[dist](num_blobs, self.rng, free_param=free_parameter)

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
            End time of the simulation. Blob arrival times are sampled
            uniformly in [0, T).
        num_blobs : int
            Number of blobs to generate.
        blob_shape : AbstractBlobShape
            Object representing the shape of the blobs.
        t_drain : float or NDArray
            Drain time scale of the blobs (exponential decay). Either a
            scalar or an array of length Nx.

        Returns
        -------
        List[Blob]
            List of Blob objects generated for the Model.

        Raises
        ------
        TypeError
            If blob_shape is not an AbstractBlobShape instance.
        """
        if not isinstance(blob_shape, AbstractBlobShape):
            raise TypeError(
                f"blob_shape must be an AbstractBlobShape, got {type(blob_shape).__name__}."
            )

        amps = self._draw_random_variables(
            self.amplitude_dist,
            self.amplitude_parameter,
            num_blobs,
        )
        wxs = self._draw_random_variables(
            self.width_p_dist, self.width_p_parameter, num_blobs
        )
        wys = self._draw_random_variables(
            self.width_s_dist, self.width_s_parameter, num_blobs
        )
        vxs = self._draw_random_variables(
            self.velocity_x_dist, self.velocity_x_parameter, num_blobs
        )
        vys = self._draw_random_variables(
            self.velocity_y_dist, self.velocity_y_parameter, num_blobs
        )
        spxs = self._draw_random_variables(
            self.shape_param_p_dist, self.shape_param_p_parameter, num_blobs
        )
        spys = self._draw_random_variables(
            self.shape_param_s_dist, self.shape_param_s_parameter, num_blobs
        )
        # For now, only a lambda parameter is implemented
        spxs_dict = [{"lam": s} for s in spxs]
        spys_dict = [{"lam": s} for s in spys]
        posxs = np.zeros(num_blobs)
        posys = self.rng.uniform(low=0.0, high=Ly, size=num_blobs)
        t_inits = self.rng.uniform(low=0, high=T, size=num_blobs)

        blobs = [
            Blob(
                blob_id=i,
                blob_shape=blob_shape,
                amplitude=amps[i],
                width_p=wxs[i],
                width_s=wys[i],
                v_x=vxs[i],
                v_y=vys[i],
                pos_x0=posxs[i],
                pos_y0=posys[i],
                t_init=t_inits[i],
                t_drain=t_drain,
                shape_parameters_p=spxs_dict[i],
                shape_parameters_s=spys_dict[i],
                blob_alignment=self.blob_alignment,
                theta=self.theta_setter() if self.theta_setter is not None else None,
            )
            for i in range(num_blobs)
        ]

        # sort blobs by amplitude
        return sorted(blobs, key=lambda x: x.amplitude)

    def set_theta_setter(self, theta_setter):
        """
        Set a lambda function to set the value of theta (blob tilting) for each blob.
        Once set, the returned angle takes precedence and `blob_alignment` is ignored.
        Important: the blob angle is measured with respect to the x axis, not with respect to the velocity vector.
        """
        self.theta_setter = theta_setter

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
        return self.velocity_y_dist == DistributionEnum.zeros
