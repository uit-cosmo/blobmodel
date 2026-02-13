"""This module defines a Blob class and related functions for discretizing and manipulating blobs."""

import warnings
from typing import Tuple, Union, Any
from nptyping import NDArray
import numpy as np
from .blob_shape import AbstractBlobShape
import cmath


class Blob:
    """
    Class representing a single blob. It stores all blob parameters and discretizes the blob on a grid through
    the function `discretize_blob`. The contribution of a single blob to a grid defined by `x`, `y` and `t` is given by:

    .. math::
        a e^{-(t-t_k)/\\tau_\\shortparallel}\\varphi\\left( \\frac{x-v(t-t_k)}{\ell_x}, \\frac{(y-y_k)-w(t-t_k)}{\ell_y} \\right)

    Where:
        - :math:`a` is the blob amplitude, `amplitude`.
        - :math:`\ell_x` is the blob width in the propagation direction, `width_prop`.
        - :math:`\ell_y` is the blob width in the perpendicular direction, `width_perp`.
        - :math:`v` is the horizontal blob velocity.
        - :math:`w` is the vertical blob velocity.
        - :math:`t_k` is the blob arriving time at the position x=0, `t_init`.
        - :math:`\\tau_\\shortparallel` is the drainage time, `t_drain`.
        - :math:`\\varphi` is the blob pulse shape, `blob_shape`.

    Additionally, a tilt angle can be provided through `theta`.
    """

    def __init__(
        self,
        blob_id: int,
        blob_shape: AbstractBlobShape,
        amplitude: float,
        width_prop: float,
        width_perp: float,
        v_x: float,
        v_y: float,
        pos_x: float,
        pos_y: float,
        t_init: float,
        t_drain: Union[float, NDArray],
        prop_shape_parameters: Union[dict, None] = None,
        perp_shape_parameters: Union[dict, None] = None,
        blob_alignment: bool = True,
        theta: float = 0,
    ) -> None:
        """
        Initialize a single blob.

        Parameters
        ----------
        blob_id : int
            Identifier for the blob.
        blob_shape : AbstractBlobShape
            Shape of the blob.
        amplitude : float
            Amplitude of the blob.
        width_prop : float
            Width of the blob in the propagation direction.
        width_perp : float
            Width of the blob in the perpendicular direction.
        v_x : float
            Velocity of the blob in the x-direction.
        v_y : float
            Velocity of the blob in the y-direction.
        pos_x : float
            Initial position of the blob in the x-direction.
        pos_y : float
            Initial position of the blob in the y-direction.
        t_init : float
            Initial time of the blob.
        t_drain : Union[float, NDArray]
            Time scale for the blob to drain.
        prop_shape_parameters : dict
            Additional shape parameters for the propagation direction.
        perp_shape_parameters : dict
            Additional shape parameters for the perpendicular direction.
        blob_alignment : bool, optional
            If blob_alignment == True, the blob shapes are rotated in the propagation direction of the blob
            If blob_alignment == False, the blob shapes are independent of the propagation direction
        theta : float
            Blob rotation. If set to None, blobs are rotated so to be aligned with their propagation. Otherwise, theta
            sets the blob angle with respect to the x axis.

            it is computed according to blob_alignment. If set to a no None value,
        the blob alignment flag is ignored. Important: the blob angle is measured with respect to the x axis, not with
         respect to the velocity vector.

        """
        assert isinstance(blob_shape, AbstractBlobShape)

        self.int = int
        self.blob_id = blob_id
        self.blob_shape = blob_shape
        self.amplitude = amplitude
        self.width_p = width_prop  # Primary width
        self.width_s = width_perp  # Secondary width
        self.v_x = v_x
        self.v_y = v_y
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.t_init = t_init
        self.t_drain = t_drain
        self.prop_shape_parameters = (
            {} if prop_shape_parameters is None else prop_shape_parameters
        )
        self.perp_shape_parameters = (
            {} if perp_shape_parameters is None else perp_shape_parameters
        )
        self.blob_alignment = blob_alignment
        self._theta = theta
        if blob_alignment:
            self._theta = cmath.phase(self.v_x + self.v_y * 1j)

    def discretize_blob(
        self,
        x: NDArray,
        y: NDArray,
        t: NDArray,
        Ly: float,
        periodic_y: bool = False,
        one_dimensional: bool = False,
    ) -> NDArray:
        """
        Discretize blob on grid. If one_dimensional the perpendicular pulse shape is ignored.

        Parameters
        ----------
        x : NDArray
            Grid coordinates in the x-direction.
        y : NDArray
            Grid coordinates in the y-direction.
        t : NDArray
            Time coordinates.
        Ly : float
            Length of domain in the y-direction.
        periodic_y : bool, optional
            Flag indicating periodicity in the y-direction (default: False).
        one_dimensional : bool, optional
            Flag indicating a one-dimensional blob (default: False).

        Notes
        -----
        The periodicity in the y direction is implemented by first substracting the number of full domain Ly
        propagations made by the blob and by summing mirror blobs at vertical positions +-Ly.

        Returns
        -------
        discretized_blob : NDArray
            Discretized blob on a 3D array with dimensions (x, y, t).

        """
        # If one_dimensional, then Ly should be 0.
        assert (one_dimensional and Ly == 0) or not one_dimensional

        if (self.width_s > Ly / 3 or self.width_p > Ly / 3) and periodic_y:
            warnings.warn(
                "blob width big compared to Ly, mirrored blobs might become apparent."
            )

        if not periodic_y or one_dimensional:
            return self._single_blob(
                x, y, t, Ly, periodic_y, one_dimensional=one_dimensional
            )

        time = t if type(t) in [int, float] else t[0][0]
        vertical_prop = self.v_y * time + self.pos_y
        number_of_y_propagations = vertical_prop // Ly

        # Sum of a centered blob is two "ghost blobs" at vertical positions +-Ly.
        return (
            self._single_blob(
                x,
                y,
                t,
                Ly,
                periodic_y,
                number_of_y_propagations,
                one_dimensional=one_dimensional,
            )
            + self._single_blob(
                x,
                y + Ly,
                t,
                Ly,
                periodic_y,
                number_of_y_propagations,
                one_dimensional=one_dimensional,
            )
            + self._single_blob(
                x,
                y - Ly,
                t,
                Ly,
                periodic_y,
                number_of_y_propagations,
                one_dimensional=one_dimensional,
            )
        )

    def _single_blob(
        self,
        x: Union[int, NDArray],
        y: Union[int, NDArray],
        t: Union[int, NDArray],
        Ly: float,
        periodic_y: bool,
        number_of_y_propagations: Union[NDArray, int] = 0,
        one_dimensional: bool = False,
    ) -> NDArray:
        """
        Calculate the discretized blob for a single blob instance.

        Parameters
        ----------
        x : NDArray
            x-coordinate
        y : NDArray
            y-coordinate
        t : NDArray
            Time coordinates.
        Ly : float
            Length of domain in the y-direction.
        periodic_y : bool
            Flag indicating periodicity in the y-direction.
        number_of_y_propagations : NDArray, optional
            Number of times the blob propagates through the domain in y-direction (default: 0).
        one_dimensional : bool, optional
            Flag indicating a one-dimensional blob (default: False).

        Returns
        -------
        blob : NDArray
            Discretized blob.

        """
        # Blob position
        pos_x = self._blob_trajectory_x(t)
        pos_y = self._blob_trajectory_y(t)
        if periodic_y:
            pos_y -= number_of_y_propagations * Ly

        # Blob frame coordinates
        xb = np.cos(self._theta) * (x - pos_x) + np.sin(self._theta) * (y - pos_y)
        yb = -np.sin(self._theta) * (x - pos_x) + np.cos(self._theta) * (y - pos_y)

        theta_x = xb / self.width_p
        theta_y = yb / self.width_s
        primary_axis_shape = self.blob_shape.get_blob_shape_prop(
            theta_x, **self.prop_shape_parameters
        )

        secondary_axis_shape = (
            1
            if one_dimensional
            else self.blob_shape.get_blob_shape_perp(
                theta_y, **self.perp_shape_parameters
            )
        )

        return (
            self.amplitude * self._drain(t) * primary_axis_shape * secondary_axis_shape
        )

    def _drain(self, t: Union[int, NDArray]) -> NDArray:
        """
        Calculate the drain factor for the blob.

        Parameters
        ----------
        t : NDArray
            Time coordinates.

        Returns
        -------
        drain_factor : NDArray
            Drain factor.

        """
        if isinstance(self.t_drain, (int, float)):
            return np.exp(-(t - self.t_init) / float(self.t_drain))
        return np.exp(-(t - self.t_init) / self.t_drain[np.newaxis, :, np.newaxis])

    def _blob_trajectory_x(self, t: Union[int, NDArray]) -> Any:
        """
        Position of the blob in the x-direction at a given time t.
        """
        return self.pos_x + self.v_x * (t - self.t_init)

    def _blob_trajectory_y(self, t: Union[int, NDArray]) -> Any:
        """
        Position of the blob in the y-direction at a given time t.
        """
        return self.pos_y + self.v_y * (t - self.t_init)
