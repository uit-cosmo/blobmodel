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
    ) -> List[Blob]:
        """
        Creates a list of Blobs used in Model.

        Notes
        -----
        - Blob draining is owned by the factory, not the `Model`: each `Blob`
          carries its own `t_drain` (`DefaultBlobFactory` takes it as a
          constructor argument, `np.inf` — no draining — by default).
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


class BlobListFactory(BlobFactory):
    """BlobFactory that returns a pre-built list of blobs.

    Use this (typically through `Model.from_blobs`) when the blobs are
    constructed by hand instead of sampled from distributions. All
    `sample_blobs` arguments (`Ly`, `T`, `num_blobs`, `blob_shape`) are
    ignored: each `Blob` already carries its own parameters.
    """

    def __init__(self, blobs: List[Blob]) -> None:
        """
        Initialize the factory with the list of blobs to realize.

        Parameters
        ----------
        blobs : List[Blob]
            Blobs returned by every `sample_blobs` call.
        """
        self._blobs = list(blobs)

    def sample_blobs(
        self,
        Ly: float,
        T: float,
        num_blobs: int,
        blob_shape: AbstractBlobShape,
    ) -> List[Blob]:
        """Return the stored blob list. All arguments are ignored."""
        return list(self._blobs)

    def is_one_dimensional(self) -> bool:
        """
        Returns True if all stored blobs have zero perpendicular velocity.

        Returns
        -------
        bool
            True if `v_y == 0` for every blob, False otherwise.
        """
        return all(blob.v_y == 0 for blob in self._blobs)


class CallableBlobFactory(BlobFactory):
    """BlobFactory that builds each blob by calling a user-provided getter.

    The getter receives the factory's random number generator, so blobs
    sampled through this factory are reproducible via
    `CallableBlobFactory(seed=...)` or `Model(seed=...)` — provided the getter
    draws its random numbers from the generator it is given instead of the
    global `np.random` state.
    """

    def __init__(
        self,
        blob_getter: Callable[[np.random.Generator], Blob],
        one_dimensional: bool = False,
        seed: Union[int, np.random.Generator, None] = None,
    ) -> None:
        """
        Initialize the factory with a blob getter.

        Parameters
        ----------
        blob_getter : Callable[[np.random.Generator], Blob]
            Function called once per blob with the factory's random number
            generator; must return a `Blob`.
        one_dimensional : bool, optional
            Whether the blobs produced by the getter are compatible with a
            one-dimensional model (i.e. have `v_y == 0`). Cannot be inferred
            from the getter, so it must be declared. By default False.
        seed : int, np.random.Generator or None, optional
            Seed (or an already constructed `numpy.random.Generator`) for the
            generator passed to `blob_getter`. A seed passed to `Model` takes
            precedence: it replaces this factory's generator via `set_rng`.
            By default None, i.e. a freshly seeded generator
            (non-reproducible).
        """
        self._blob_getter = blob_getter
        self._one_dimensional = one_dimensional
        self.rng = np.random.default_rng(seed)

    def sample_blobs(
        self,
        Ly: float,
        T: float,
        num_blobs: int,
        blob_shape: AbstractBlobShape,
    ) -> List[Blob]:
        """
        Create `num_blobs` blobs by calling the getter with `self.rng`.

        `Ly`, `T` and `blob_shape` are ignored: the getter is expected to
        fully specify each blob.
        """
        return [self._blob_getter(self.rng) for _ in range(num_blobs)]

    def is_one_dimensional(self) -> bool:
        """Return the `one_dimensional` flag declared at construction."""
        return self._one_dimensional


ParameterSampler = Callable[[np.random.Generator, int], np.ndarray]
"""Signature of a custom sampler for `DefaultBlobFactory.set_sampler`: called
with the factory's random number generator and the number of blobs, returns
one sampled value per blob."""


class DefaultBlobFactory(BlobFactory):
    """Default implementation of BlobFactory.

    Samples each blob parameter independently. Every parameter defaults to a
    degenerate (constant) distribution except the amplitude, which is
    exponential with mean 1 (the canonical FPP choice), and the perpendicular
    velocity ``vy``, which is zero (matching the `Blob` default ``v_y=0``, and
    making the default factory compatible with one-dimensional models).
    Individual parameters are reconfigured with `set_sampler`, which accepts
    either a `DistributionEnum` or an arbitrary sampling callable.
    """

    # Configurable parameter names -> (default distribution, free parameter).
    _DEFAULT_SAMPLERS = {
        "amplitude": (DistributionEnum.exp, 1.0),
        "wp": (DistributionEnum.deg, 1.0),
        "ws": (DistributionEnum.deg, 1.0),
        "vx": (DistributionEnum.deg, 1.0),
        "vy": (DistributionEnum.zeros, 0.0),
        "spp": (DistributionEnum.deg, 0.5),
        "sps": (DistributionEnum.deg, 0.5),
    }

    def __init__(
        self,
        t_drain: Union[float, NDArray, int] = np.inf,
        blob_alignment: bool = False,
        seed: Union[int, np.random.Generator, None] = None,
    ) -> None:
        """
        Default implementation of BlobFactory.

        Blob parameter distributions are configured with `set_sampler`; the
        defaults are an exponential amplitude with mean 1, zero perpendicular
        velocity ``vy``, and degenerate (constant) distributions for
        everything else: widths and ``vx`` 1, shape parameters 0.5.

        .. versionchanged:: 2.0.0
            ``vy`` previously defaulted to the constant 1, so a bare factory
            produced diagonally propagating blobs (inconsistent with the
            `Blob` default ``v_y=0``) and was not one-dimensional-compatible.

        Parameters
        ----------
        t_drain : float or array-like, optional
            Drain time scale of the blobs (exponential decay), applied to every
            sampled blob. Can be a single value or an array-like of length Nx
            (the number of grid points of the model's geometry in the
            x-direction). By default `np.inf`, i.e. no draining.
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

        Raises
        ------
        ValueError
            If `t_drain` is not positive.
        """
        if np.any(np.asarray(t_drain) <= 0):
            raise ValueError(f"t_drain must be positive, got t_drain = {t_drain}.")

        # Per-parameter distribution (None for custom callables) and sampler.
        self._dists: dict = {}
        self._free_parameters: dict = {}
        self._samplers: dict = {}
        for parameter, (dist, free_parameter) in self._DEFAULT_SAMPLERS.items():
            self.set_sampler(parameter, dist, free_parameter)

        self.t_drain = t_drain
        self.blob_alignment = blob_alignment
        self.theta_setter: Union[Callable[[], float], None] = None
        self.rng = np.random.default_rng(seed)

    def set_sampler(
        self,
        parameter: str,
        sampler: Union[DistributionEnum, ParameterSampler],
        free_parameter: Union[float, None] = None,
    ) -> "DefaultBlobFactory":
        """
        Configure how one blob parameter is sampled.

        Parameters
        ----------
        parameter : str
            Which blob parameter to configure. One of:
            - "amplitude": blob amplitude
            - "wp": width in the principal (propagation) blob direction
            - "ws": width in the secondary (perpendicular) blob direction
            - "vx": velocity in the x-direction
            - "vy": velocity in the y-direction
            - "spp": pulse shape parameter in the principal direction
            - "sps": pulse shape parameter in the secondary direction
        sampler : DistributionEnum or Callable[[np.random.Generator, int], np.ndarray]
            Either one of the built-in distributions (see Notes), or a callable
            drawing the values itself: it is called with the factory's random
            number generator and the number of blobs and must return one value
            per blob. Draw from the generator you are given (not the global
            `np.random` state) so realizations stay reproducible through
            `DefaultBlobFactory(seed=...)` or `Model(seed=...)`.
        free_parameter : float, optional
            Free parameter of the built-in distribution (see Notes), by
            default 1.0. Only valid together with a `DistributionEnum`.

        Returns
        -------
        DefaultBlobFactory
            The factory itself, so calls can be chained:
            ``DefaultBlobFactory().set_sampler("vy", DistributionEnum.zeros)``.

        Notes
        -----
        - The following distributions are implemented:
            - exp: exponential distribution with mean `free_parameter`
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
            If `sampler` is neither a DistributionEnum member nor a callable.
        ValueError
            If `parameter` is not one of the names listed above, if a width
            (`wp` or `ws`) is uniform with `free_parameter` > 2 (which would
            produce negative blob widths), or if `free_parameter` is combined
            with a callable sampler.
        """
        if parameter not in self._DEFAULT_SAMPLERS:
            raise ValueError(
                f"Unknown parameter '{parameter}', "
                f"must be one of {sorted(self._DEFAULT_SAMPLERS)}."
            )
        if isinstance(sampler, DistributionEnum):
            if free_parameter is None:
                free_parameter = 1.0
            if (
                parameter in ("wp", "ws")
                and sampler == DistributionEnum.uniform
                and free_parameter > 2
            ):
                raise ValueError(
                    f"free_parameter = {free_parameter} with a uniform {parameter} distribution "
                    f"would produce negative blob widths: the uniform distribution has support "
                    f"[1 - free_parameter / 2, 1 + free_parameter / 2], so free_parameter must be <= 2."
                )
            dist_function = DISTRIBUTIONS[sampler]
            self._dists[parameter] = sampler
            self._free_parameters[parameter] = free_parameter
            self._samplers[parameter] = (
                lambda rng, num_blobs, _dist_function=dist_function, _free_parameter=free_parameter: _dist_function(
                    num_blobs, rng, free_param=_free_parameter
                )
            )
        elif callable(sampler):
            if free_parameter is not None:
                raise ValueError(
                    "free_parameter only applies to DistributionEnum samplers; "
                    "a callable sampler takes no free parameter."
                )
            self._dists[parameter] = None
            self._free_parameters[parameter] = None
            self._samplers[parameter] = sampler
        else:
            raise TypeError(
                f"sampler for '{parameter}' must be a DistributionEnum or a callable, "
                f"got {type(sampler).__name__}."
            )
        return self

    def _draw_random_variables(self, parameter: str, num_blobs: int) -> np.ndarray:
        """Draw `num_blobs` values for one parameter from its sampler."""
        values = np.asarray(self._samplers[parameter](self.rng, num_blobs))
        if values.shape != (num_blobs,):
            raise ValueError(
                f"The sampler for '{parameter}' returned shape {values.shape}, "
                f"expected ({num_blobs},)."
            )
        return values

    def sample_blobs(
        self,
        Ly: float,
        T: float,
        num_blobs: int,
        blob_shape: AbstractBlobShape,
    ) -> List[Blob]:
        """
        Creates a list of Blobs used in the Model.

        Every blob is given the factory's `t_drain` (a constructor argument,
        `np.inf` — no draining — by default).

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

        amps = self._draw_random_variables("amplitude", num_blobs)
        wxs = self._draw_random_variables("wp", num_blobs)
        wys = self._draw_random_variables("ws", num_blobs)
        vxs = self._draw_random_variables("vx", num_blobs)
        vys = self._draw_random_variables("vy", num_blobs)
        spxs = self._draw_random_variables("spp", num_blobs)
        spys = self._draw_random_variables("sps", num_blobs)
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
                t_drain=self.t_drain,
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
        - Only the built-in `DistributionEnum.zeros` distribution for "vy" is
          recognized as one-dimensional; a custom callable sampler is assumed
          two-dimensional even if it always returns zeros.

        """
        return self._dists["vy"] == DistributionEnum.zeros
