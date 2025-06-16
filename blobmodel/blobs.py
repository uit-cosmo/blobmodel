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
        self.width_prop = width_prop
        self.width_perp = width_perp
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

        if (self.width_perp > Ly / 3 or self.width_prop > Ly / 3) and periodic_y:
            warnings.warn(
                "blob width big compared to Ly, mirrored blobs might become apparent."
            )

        x_perp, y_perp = self._rotate(
            origin=(self.pos_x, self.pos_y), x=x, y=y, angle=-self._theta
        )
        v_x_new = self.v_x * np.cos(self._theta) + self.v_y * np.sin(self._theta)
        v_y_new = -self.v_x * np.sin(self._theta) + self.v_y * np.cos(self._theta)
        self.v_x, self.v_y = v_x_new, v_y_new
        if not periodic_y or one_dimensional:
            return self._single_blob(
                x_perp, y_perp, t, Ly, periodic_y, one_dimensional=one_dimensional
            )
        if self._theta == 0:
            number_of_y_propagations = 0
        else:
            x_border = (Ly - self.pos_y) / np.sin(self._theta)
            adjusted_Ly = Ly / np.sin(self._theta)
            prop_dir = (
                self._prop_dir_blob_position(t)
                if type(t) in [int, float]  # t has dimensionality = 0, used for testing
                else self._prop_dir_blob_position(t[0, 0])
            )
            number_of_y_propagations = (
                prop_dir + adjusted_Ly - x_border
            ) // adjusted_Ly
        return (
            self._single_blob(
                x_perp,
                y_perp,
                t,
                Ly,
                periodic_y,
                number_of_y_propagations,
                one_dimensional=one_dimensional,
            )
            + self._single_blob(
                x_perp,
                y_perp,
                t,
                Ly,
                periodic_y,
                number_of_y_propagations,
                x_offset=Ly * np.sin(self._theta),
                y_offset=Ly * np.cos(self._theta),
                one_dimensional=one_dimensional,
            )
            + self._single_blob(
                x_perp,
                y_perp,
                t,
                Ly,
                periodic_y,
                number_of_y_propagations,
                x_offset=-Ly * np.sin(self._theta),
                y_offset=-Ly * np.cos(self._theta),
                one_dimensional=one_dimensional,
            )
        )

    def _single_blob(
        self,
        x_prop: Union[int, NDArray],
        y_perp: Union[int, NDArray],
        t: Union[int, NDArray],
        Ly: float,
        periodic_y: bool,
        number_of_y_propagations: Union[NDArray, int] = 0,
        x_offset: Union[NDArray, int] = 0,
        y_offset: Union[NDArray, int] = 0,
        one_dimensional: bool = False,
    ) -> NDArray:
        """
        Calculate the discretized blob for a single blob instance.

        Parameters
        ----------
        x_prop : NDArray
            Propagation direction coordinates.
        y_perp : NDArray
            Perpendicular coordinates.
        t : NDArray
            Time coordinates.
        Ly : float
            Length of domain in the y-direction.
        periodic_y : bool
            Flag indicating periodicity in the y-direction.
        number_of_y_propagations : NDArray, optional
            Number of times the blob propagates through the domain in y-direction (default: 0).
        x_offset : NDArray, optional
            Offset in the x-direction (default: 0).
        y_offset : NDArray, optional
            Offset in the y-direction (default: 0).
        one_dimensional : bool, optional
            Flag indicating a one-dimensional blob (default: False).

        Returns
        -------
        blob : NDArray
            Discretized blob.

        """
        return (
            self.amplitude
            * self._drain(t)
            * self._propagation_direction_shape(
                x_prop + x_offset,
                t,
                Ly,
                periodic_y,
                number_of_y_propagations=number_of_y_propagations,
            )
            * (
                1
                if one_dimensional
                else self._perpendicular_direction_shape(
                    y_perp + y_offset,
                    t,
                    Ly,
                    periodic_y,
                    number_of_y_propagations=number_of_y_propagations,
                )
            )
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

    def _propagation_direction_shape(
        self,
        x: Union[int, NDArray],
        t: Union[int, NDArray],
        Ly: float,
        periodic_y: bool,
        number_of_y_propagations: Union[int, NDArray],
    ) -> NDArray:
        """
        Calculate the shape in the propagation direction.

        Parameters
        ----------
        x : NDArray
            Coordinates in the x-direction.
        t : NDArray
            Time coordinates.
        Ly : float
            Length of domain in the y-direction.
        periodic_y : bool
            Flag indicating periodicity in the y-direction.
        number_of_y_propagations : NDArray
            Number of times the blob propagates through the domain in y-direction.

        Returns
        -------
        shape : NDArray
            Shape in the propagation direction.

        """
        if periodic_y:
            x_diffs = (
                x
                - self._prop_dir_blob_position(t)
                + number_of_y_propagations * Ly * np.sin(self._theta)
            )
        else:
            x_diffs = x - self._prop_dir_blob_position(t)
        theta_x = x_diffs / self.width_prop
        return self.blob_shape.get_blob_shape_prop(
            theta_x, **self.prop_shape_parameters
        )

    def _perpendicular_direction_shape(
        self,
        y: Union[int, NDArray],
        t: Union[int, NDArray],
        Ly: float,
        periodic_y: bool,
        number_of_y_propagations: Union[int, NDArray],
    ) -> NDArray:
        """
        Calculate the shape in the perpendicular direction.

        Parameters
        ----------
        y : NDArray
            Coordinates in the y-direction.
        Ly : float
            Length of domain in the y-direction.
        periodic_y : bool
            Flag indicating periodicity in the y-direction.
        number_of_y_propagations : NDArray
            Number of times the blob propagates through the domain in y-direction.

        Returns
        -------
        shape : NDArray
            Blob shape in the perpendicular direction.

        """
        if periodic_y:
            y_diffs = (
                y
                - self._perp_dir_blob_position(t)
                + number_of_y_propagations * Ly * np.cos(self._theta)
            )
        else:
            y_diffs = y - self._perp_dir_blob_position(t)
        theta_y = y_diffs / self.width_perp
        return self.blob_shape.get_blob_shape_perp(
            theta_y, **self.perp_shape_parameters
        )

    def _prop_dir_blob_position(self, t: Union[int, NDArray]) -> Any:
        """
        Calculate the position of the blob in the propagation direction.

        Parameters
        ----------
        t : NDArray
            Time coordinates.

        Returns
        -------
        position : NDArray
            Position of the blob in the propagation direction.

        """
        return self.pos_x + self.v_x * (t - self.t_init)

    def _perp_dir_blob_position(self, t: Union[int, NDArray]) -> Any:
        """
        Return the position of the blob in the perpendicular direction.

        Parameters
        ----------
        t : NDArray
            Time coordinates.

        Returns
        -------
        position : float
            Position of the blob in the perpendicular direction.

        """
        return self.pos_y + self.v_y * (t - self.t_init)

    def _rotate(
        self, origin: Tuple[float, float], x: NDArray, y: NDArray, angle: float
    ) -> Tuple[NDArray, NDArray]:
        """
        Rotate the coordinates around a given origin point.

        Parameters
        ----------
        origin : Tuple[float, float]
            Origin point of rotation.
        x : NDArray
            Coordinates in the x-direction.
        y : NDArray
            Coordinates in the y-direction.
        angle : float
            Rotation angle.

        Returns
        -------
        rotated_coordinates : Tuple[NDArray, NDArray]
            Rotated coordinates.

        """
        ox, oy = origin
        px, py = x, y

        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy
