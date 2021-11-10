from typing import Any
from nptyping import NDArray, Float
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
        Attributes
        ----------
        Nx: int, grid points in x
        Ny: int, grid points in y
        Lx: float, length of grid in x
        Ly: float, length of grid in y
        dt: float, time step
        T: float, time length
        periodic_y: bool, optional
            allow periodicity in y-direction
        """
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.dt = dt
        self.T = T
        self.periodic_y = periodic_y

        # calculate x, y and t coordinates
        self.x: NDArray[Any, Float[64]] = np.arange(0, self.Lx, self.Lx / self.Nx)
        if self.Ly == 0:
            self.y: NDArray[Any, Float[64]] = 0
        else:
            self.y = np.arange(0, self.Ly, self.Ly / self.Ny)
        self.t: NDArray[Any, Float[64]] = np.arange(0, self.T, self.dt)
        self.x_matrix, self.y_matrix, self.t_matrix = np.meshgrid(
            self.x, self.y, self.t
        )

    def __str__(self) -> str:
        """string representation of Geometry."""
        return (
            f"Geometry parameters:  Nx:{self.Nx},  Ny:{self.Ny}, Lx:{self.Lx}, Ly:{self.Ly}, "
            + f"dt:{self.dt}, T:{self.T}, y-periodicity:{self.periodic_y}"
        )
