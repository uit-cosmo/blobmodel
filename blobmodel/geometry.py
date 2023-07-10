"""This module defines the Geometry class for creating a grid for the Model."""

from typing import Any
from nptyping import NDArray
from typing_extensions import Literal
import numpy as np


class Geometry:
    """Define grid for Model."""

    def __init__(
        self,
        Nx: int,
        Ny: int,
        Lx: float,
        Ly: float,
        dt: float,
        T: float,
        periodic_y: bool,
    ) -> None:
        """
        Initialize a Geometry object.

        Parameters
        ----------
        Nx : int
            Number of grid points in the x-direction.
        Ny : int
            Number of grid points in the y-direction.
        Lx : float
            Length of domain in the x-direction.
        Ly : float
            Length of domain in the y-direction.
        dt : float
            Time step.
        T : float
            Time length.
        periodic_y : bool
            Flag indicating whether periodicity is allowed in the y-direction.
        """
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.dt = dt
        self.T = T
        self.periodic_y = periodic_y

        # calculate x, y and t coordinates
        self.x: NDArray[Literal[64], Any] = np.arange(0, self.Lx, self.Lx / self.Nx)
        if self.Ly == 0:
            self.y: NDArray[Literal[64], Any] = np.array([0])
        else:
            self.y = np.arange(0, self.Ly, self.Ly / self.Ny)
        self.t: NDArray[Literal[64], Any] = np.arange(0, self.T, self.dt)
        self.x_matrix, self.y_matrix, self.t_matrix = np.meshgrid(
            self.x, self.y, self.t
        )

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
            + f"dt:{self.dt}, T:{self.T}, y-periodicity:{self.periodic_y}"
        )
