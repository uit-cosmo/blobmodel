"""This module defines a Blob class and related functions for discretizing and manipulating blobs."""

from typing import Union, Any, Optional
from nptyping import NDArray
import numpy as np
from .blob_shape import AbstractBlobShape, BlobShapeImpl
import cmath


class Blob:
    """
    Class representing a single blob. It stores all blob parameters and discretizes the blob on a grid through
    the function `discretize_blob`. The contribution of a single blob to a grid defined by `x`, `y` and `t` is given by:

    .. math::
        a e^{-(t-t_k)/\\tau_\\shortparallel}\\varphi\\left( \\frac{x-v(t-t_k)}{\ell_x}, \\frac{(y-y_k)-w(t-t_k)}{\ell_y} \\right)

    Where:
        - :math:`a` is the blob amplitude, `amplitude`.
        - :math:`\ell_x` is the blob width in the principal direction, `width_p`.
        - :math:`\ell_y` is the blob width in the secondary direction, `width_s`.
        - :math:`v` is the horizontal blob velocity.
        - :math:`w` is the vertical blob velocity.
        - :math:`t_k` is the blob arriving time at the position x=0, `t_init`.
        - :math:`\\tau_\\shortparallel` is the drainage time, `t_drain`.
        - :math:`\\varphi` is the blob pulse shape, `blob_shape`.

    Additionally, a tilt angle can be provided through `theta`.
    """

    def __init__(
        self,
        blob_id: int = 0,
        blob_shape: Optional[AbstractBlobShape] = None,
        amplitude: float = 1.0,
        width_p: float = 1.0,
        width_s: float = 1.0,
        v_x: float = 1.0,
        v_y: float = 0.0,
        pos_x0: float = 0.0,
        pos_y0: float = 0.0,
        t_init: float = 0.0,
        t_drain: Union[float, NDArray] = np.inf,
        shape_parameters_p: Union[dict, None] = None,
        shape_parameters_s: Union[dict, None] = None,
        blob_alignment: bool = False,
        theta: Optional[float] = None,
    ) -> None:
        """
        Initialize a single blob.

        Parameters
        ----------
        blob_id : int, optional
            Identifier for the blob (default 0). Purely metadata: blob labels
            in ``Model`` (``labels="individual"``) are assigned from each
            blob's position in the factory output, not from ``blob_id``.
        blob_shape : AbstractBlobShape, optional
            Shape of the blob. Default None, which means a Gaussian shape in
            both directions, ``BlobShapeImpl()``.
        amplitude : float, optional
            Amplitude of the blob. Default 1.
        width_p : float, optional
            Width of the blob in the propagation direction. Default 1.
        width_s : float, optional
            Width of the blob in the perpendicular direction. Default 1.
        v_x : float, optional
            Velocity of the blob in the x-direction. Default 1.
        v_y : float, optional
            Velocity of the blob in the y-direction. Default 0.
        pos_x0 : float, optional
            Initial position of the blob in the x-direction. Default 0.
        pos_y0 : float, optional
            Initial position of the blob in the y-direction. Default 0.
        t_init : float, optional
            Initial time of the blob. Default 0.
        t_drain : Union[float, NDArray], optional
            Time scale for the blob to drain. Default ``np.inf`` = no
            draining.
        shape_parameters_p : dict
            Additional shape parameters for the propagation direction.
        shape_parameters_s : dict
            Additional shape parameters for the perpendicular direction.
        blob_alignment : bool, optional
            Only used when ``theta is None``. If True, the blob shape is rotated to
            align with the blob propagation direction; if False, the blob shape is
            axis-aligned (no rotation). Default False.
        theta : float or None, optional
            Blob rotation angle with respect to the x-axis (not the velocity
            vector). If not None, this angle is used directly and ``blob_alignment``
            is ignored. If None (the default), the angle is determined by
            ``blob_alignment``: the velocity phase when it is True, or 0 when False.

        Raises
        ------
        TypeError
            If ``blob_shape`` is not an ``AbstractBlobShape`` instance.
        ValueError
            If ``width_p`` or ``width_s`` is not positive, or if ``t_drain``
            is not positive (every element, when it is an array).

        """
        if blob_shape is None:
            blob_shape = BlobShapeImpl()
        if not isinstance(blob_shape, AbstractBlobShape):
            raise TypeError(
                f"blob_shape must be an AbstractBlobShape, got {type(blob_shape).__name__}."
            )
        if width_p <= 0 or width_s <= 0:
            raise ValueError(
                f"Blob widths must be positive, got width_p = {width_p} and width_s = {width_s}."
            )
        if np.any(np.asarray(t_drain) <= 0):
            raise ValueError(f"t_drain must be positive, got t_drain = {t_drain}.")

        self.blob_id = blob_id
        self.blob_shape = blob_shape
        self.amplitude = amplitude
        self.width_p = width_p  # Primary width
        self.width_s = width_s  # Secondary width
        self.v_x = v_x
        self.v_y = v_y
        self.pos_x0 = pos_x0
        self.pos_y0 = pos_y0
        self.t_init = t_init
        self.t_drain = t_drain
        self.shape_parameters_p = (
            {} if shape_parameters_p is None else shape_parameters_p
        )
        self.shape_parameters_s = (
            {} if shape_parameters_s is None else shape_parameters_s
        )
        self.blob_alignment = blob_alignment
        if theta is not None:
            self._theta = theta
        elif blob_alignment:
            self._theta = cmath.phase(self.v_x + self.v_y * 1j)
        else:
            self._theta = 0

    def discretize_blob(
        self,
        x: NDArray,
        y: NDArray,
        t: NDArray,
        Ly: float,
        periodic_y: bool = False,
        one_dimensional: bool = False,
        y0: float = 0,
    ) -> NDArray:
        """
        Discretize blob on grid. If one_dimensional the secondary pulse shape is ignored.

        Parameters
        ----------
        x : NDArray
            Grid coordinates in the x-direction, as an array broadcastable to
            shape (Ny, Nx, Nt): either a full meshgrid array or a 1D
            coordinate array reshaped as ``x[np.newaxis, :, np.newaxis]``.
        y : NDArray
            Grid coordinates in the y-direction, as an array broadcastable to
            shape (Ny, Nx, Nt): either a full meshgrid array or a 1D
            coordinate array reshaped as ``y[:, np.newaxis, np.newaxis]``.
        t : NDArray
            Time coordinates, as an array broadcastable to shape (Ny, Nx, Nt):
            either a full meshgrid array or a 1D coordinate array reshaped as
            ``t[np.newaxis, np.newaxis, :]``. A scalar t (single time point)
            is also accepted.
        Ly : float
            Length of domain in the y-direction.
        periodic_y : bool, optional
            Flag indicating periodicity in the y-direction (default: False).
        one_dimensional : bool, optional
            Flag indicating a one-dimensional blob (default: False).
        y0 : float, optional
            Origin of the domain in the y-direction (default: 0). Only used
            when ``periodic_y`` is True, where the blob position is wrapped
            into the domain ``[y0, y0 + Ly)``.

        Notes
        -----
        The periodicity in the y direction is implemented by first substracting the number of full domain Ly
        propagations made by the blob and by summing mirror blobs at vertical positions +-Ly.

        Returns
        -------
        discretized_blob : NDArray
            Discretized blob on a 3D array with dimensions (y, x, t),
            i.e. shape (Ny, Nx, Nt).

        Raises
        ------
        ValueError
            If ``one_dimensional`` is True and ``Ly`` is not 0.

        """
        if one_dimensional and Ly != 0:
            raise ValueError(f"One dimensional blobs require Ly == 0, got Ly = {Ly}.")

        if not periodic_y or one_dimensional:
            return self._single_blob(
                x, y, t, Ly, periodic_y, one_dimensional=one_dimensional
            )

        time = t if np.ndim(t) == 0 else t[0][0]
        vertical_prop = self.v_y * (time - self.t_init) + self.pos_y0
        # Wrap the blob position into the domain [y0, y0 + Ly).
        number_of_y_propagations = (vertical_prop - y0) // Ly

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
        primary_axis_shape = self.blob_shape.get_blob_shape_p(
            theta_x, **self.shape_parameters_p
        )

        secondary_axis_shape = (
            1
            if one_dimensional
            else self.blob_shape.get_blob_shape_s(theta_y, **self.shape_parameters_s)
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
        return self.pos_x0 + self.v_x * (t - self.t_init)

    def _blob_trajectory_y(self, t: Union[int, NDArray]) -> Any:
        """
        Position of the blob in the y-direction at a given time t.
        """
        return self.pos_y0 + self.v_y * (t - self.t_init)
