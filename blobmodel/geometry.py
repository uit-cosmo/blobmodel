"""This module defines the Geometry class for creating a grid for the Model."""

from typing import Any, Literal
from nptyping import NDArray
import numpy as np


class Geometry:
    """
    Represents the space and time grid used by the Model class to discretize the blob evolution.
    It stores one 1D coordinate array per dimension (`x`, `y` and `t`) based on the desired
    resolution and length in each coordinate; the Model broadcasts them against each other
    instead of materializing full (Ny, Nx, Nt) meshgrids.
    """

    def __init__(
        self,
        Nx: int = 100,
        Ny: int = 100,
        Lx: float = 10,
        Ly: float = 10,
        dt: float = 0.1,
        T: float = 10,
        t_init: float = 0,
        periodic_y: bool = False,
        x0: float = 0,
        y0: float = 0,
    ) -> None:
        """
        Initialize a Geometry object.

        Parameters
        ----------
        Nx : int, optional
            Number of grid points in the x-direction.
        Ny : int, optional
            Number of grid points in the y-direction.
        Lx : float, optional
            Length of domain in the x-direction.
        Ly : float, optional
            Length of domain in the y-direction.
        dt : float, optional
            Time step.
        T : float, optional
            End time of the simulation. The time grid is
            ``np.arange(t_init, T, dt)``, so the realized time length is
            ``T - t_init``. Note that ``t_init`` may be negative, e.g.
            ``t_init=-T`` gives a time grid centered on 0.
        t_init : float, optional
            Initial time.
        periodic_y : bool, optional
            Flag indicating whether periodicity is allowed in the y-direction.
        x0 : float, optional
            Origin of the domain in the x-direction: the x grid is
            ``np.linspace(x0, x0 + Lx, Nx, endpoint=False)``. Blob positions
            (``pos_x0``) are absolute coordinates — offsetting the domain does
            not shift the blobs, it moves the observation window. Note that
            `DefaultBlobFactory` seeds blobs at ``pos_x0=0`` regardless of
            ``x0``.
        y0 : float, optional
            Origin of the domain in the y-direction, analogous to ``x0``. Note
            that `DefaultBlobFactory` samples blob positions ``pos_y0`` in
            ``[0, Ly]`` regardless of ``y0``.
        """
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.dt = dt
        self.T = T
        self.t_init = t_init
        self.periodic_y = periodic_y
        self.x0 = x0
        self.y0 = y0

        # calculate x, y and t coordinates
        self.x: NDArray[Literal[64], Any] = np.linspace(
            self.x0, self.x0 + self.Lx, num=self.Nx, endpoint=False
        )
        if self.Ly == 0:
            self.y: NDArray[Literal[64], Any] = np.array([self.y0], dtype="float64")
        else:
            self.y = np.linspace(
                self.y0, self.y0 + self.Ly, num=self.Ny, endpoint=False
            )
        self.t: NDArray[Literal[64], Any] = np.arange(t_init, self.T, self.dt)

    @classmethod
    def from_arrays(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        periodic_y: bool = False,
    ) -> "Geometry":
        """
        Build a Geometry from explicit coordinate arrays.

        The grid invariants (`Nx`, `Lx`, `x0`, `dt`, ...) are derived from the
        arrays instead of being provided by the caller. All arrays must be
        uniformly spaced — the Model's `speed_up` truncation and the blob
        discretization assume it.

        Parameters
        ----------
        x : np.ndarray
            1D array of x coordinates, uniformly spaced.
        y : np.ndarray
            1D array of y coordinates, uniformly spaced. A single-element
            array yields a one-dimensional (Ly = 0) geometry.
        t : np.ndarray
            1D array of time coordinates, uniformly spaced.
        periodic_y : bool, optional
            Flag indicating whether periodicity is allowed in the y-direction.

        Returns
        -------
        Geometry
            Geometry whose coordinate arrays match the inputs. The domain
            lengths follow the ``endpoint=False`` convention used by
            `Geometry.__init__`, i.e. ``Lx = dx * Nx``.

        Raises
        ------
        ValueError
            If any array is not 1D, is empty, is not uniformly spaced, or is
            not strictly increasing.
        """
        x, y, t = np.asarray(x), np.asarray(y), np.asarray(t)
        for name, arr in (("x", x), ("y", y), ("t", t)):
            if arr.ndim != 1 or arr.size == 0:
                raise ValueError(f"{name} must be a non-empty 1D array.")
            spacings = np.diff(arr)
            if arr.size > 1 and not np.allclose(
                spacings, spacings[0], rtol=1e-8, atol=0
            ):
                raise ValueError(f"{name} must be uniformly spaced.")
            # A descending array passes the uniform-spacing check but yields
            # a negative domain length, silently breaking the speed_up
            # truncation windows.
            if arr.size > 1 and spacings[0] <= 0:
                raise ValueError(f"{name} must be strictly increasing.")

        dx = x[1] - x[0] if x.size > 1 else 1.0
        dy = y[1] - y[0] if y.size > 1 else 0.0
        dt = t[1] - t[0] if t.size > 1 else 1.0
        geometry = cls(
            Nx=x.size,
            Ny=y.size,
            Lx=dx * x.size,
            Ly=dy * y.size,
            dt=dt,
            T=t[-1] + dt,
            t_init=t[0],
            periodic_y=periodic_y,
            x0=x[0],
            y0=y[0],
        )
        # Keep the caller's exact arrays: re-generating them from the derived
        # invariants (in particular np.arange for t) is subject to
        # floating-point off-by-one at the endpoints.
        geometry.x = x.astype("float64")
        geometry.y = y.astype("float64")
        geometry.t = t.astype("float64")
        return geometry

    def __str__(self) -> str:
        """
        Return a string representation of the Geometry object.

        Returns
        -------
        str
            String representation of the Geometry object.
        """
        return (
            f"Geometry parameters:  Nx:{self.Nx},  Ny:{self.Ny}, Lx:{self.Lx}, Ly:{self.Ly}, "
            + f"dt:{self.dt}, T:{self.T}, t_init:{self.t_init}, y-periodicity:{self.periodic_y}, "
            + f"x0:{self.x0}, y0:{self.y0}"
        )
