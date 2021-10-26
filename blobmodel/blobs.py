from dataclasses import dataclass
import numpy as np
from nptyping import NDArray
from typing import Any, Tuple


@dataclass
class Blob:
    """
    A single blob
    """

    id: int
    blob_shape: str
    amplitude: float
    width_x: float
    width_y: float
    v_x: float
    v_y: float
    pos_x: float
    pos_y: float
    t_init: float
    t_drain: float

    def discretize_blob(
        self,
        x: NDArray,
        y: NDArray,
        t: NDArray,
        periodic_y: bool = False,
        Ly: float = 0,
    ) -> NDArray:
        """
        Discretize blob on grid
        The following blob shapes are implemented:
                gauss: 2D gaussian function
                exp: one sided exponential in x and gaussian in y

                Returns
                -------
                discretized blob on 3d array with dimensions x,y and t : np.array
        """
        if self.v_x != 0:
            self.__theta = np.arctan(self.v_y / self.v_x)
        else:
            self.__theta = np.pi / 2 * np.sign(self.v_y)

        x_perp, y_perp = self.__rotate(
            origin=(self.pos_x, self.pos_y), x=x, y=y, angle=-self.__theta
        )
        if periodic_y:
            return (
                self.amplitude
                * self.__drain(t)
                * self.__porpagation_direction_shape(x_perp, t)
                * self.__perpendicular_direction_shape(y_perp, t)
                + self.amplitude
                * self.__drain(t)
                * self.__porpagation_direction_shape(
                    x_perp + Ly * np.sin(self.__theta), t
                )
                * self.__perpendicular_direction_shape(
                    y_perp + Ly * np.cos(self.__theta), t
                )
                * self.__blob_arrival(t)
                + self.amplitude
                * self.__drain(t)
                * self.__porpagation_direction_shape(
                    x_perp - Ly * np.sin(self.__theta), t
                )
                * self.__perpendicular_direction_shape(
                    y_perp - Ly * np.cos(self.__theta), t
                )
                * self.__blob_arrival(t)
            )
        else:
            return (
                self.amplitude
                * self.__drain(t)
                * self.__porpagation_direction_shape(x_perp, t)
                * self.__perpendicular_direction_shape(y_perp, t)
                * self.__blob_arrival(t)
            )

    def __drain(self, t: NDArray) -> NDArray:
        return np.exp(-(t - self.t_init) / self.t_drain)

    def __blob_arrival(self, t: NDArray) -> NDArray:
        return np.heaviside(t - self.t_init, 1)

    def __porpagation_direction_shape(self, x: NDArray, t: NDArray) -> NDArray:
        x_diffs = x - self.__prop_dir_blob_position(t)

        if self.blob_shape == "gauss":
            return 1 / np.sqrt(np.pi) * np.exp(-(x_diffs ** 2 / self.width_x ** 2))
        elif self.blob_shape == "exp":
            return np.exp(x_diffs) * np.heaviside(-1.0 * (x_diffs), 1)
        else:
            raise NotImplementedError(
                self.__class__.__name__ + ".blob shape not implemented"
            )

    def __perpendicular_direction_shape(self, y: NDArray, t: NDArray) -> NDArray:
        y_diffs = y - self.__perp_dir_blob_position(t)
        return 1 / np.sqrt(np.pi) * np.exp(-(y_diffs ** 2) / self.width_y ** 2)

    def __prop_dir_blob_position(self, t: NDArray) -> NDArray:
        return self.pos_x + (self.v_x ** 2 + self.v_y ** 2) ** 0.5 * (t - self.t_init)

    def __perp_dir_blob_position(self, t: NDArray) -> NDArray:
        return self.pos_y

    def __rotate(
        self, origin: Tuple, x: NDArray, y: NDArray, angle: float
    ) -> Tuple[Any, Any]:
        ox, oy = origin
        px, py = x, y

        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy
