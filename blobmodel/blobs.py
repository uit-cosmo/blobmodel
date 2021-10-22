from dataclasses import dataclass
import numpy as np
from nptyping import NDArray


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
        return (
            self.amplitude
            * self.__drain(t)
            * self.__x_shape(x, t)
            * self.__y_shape(y, t, periodic_y, Ly)
            * self.__blob_arrival(t)
        )

    def __drain(self, t: NDArray) -> NDArray:
        return np.exp(-(t - self.t_init) / self.t_drain)

    def __blob_arrival(self, t: NDArray) -> NDArray:
        return np.heaviside(t - self.t_init, 1)

    def __x_shape(self, x: NDArray, t: NDArray) -> NDArray:
        if self.blob_shape == "gauss":
            return (
                1
                / np.sqrt(np.pi)
                * np.exp(-((x - self.__get_x_blob_pos(t)) ** 2 / self.width_x ** 2))
            )
        elif self.blob_shape == "exp":
            return np.exp(x - self.__get_x_blob_pos(t)) * np.heaviside(
                -1.0 * (x - self.__get_x_blob_pos(t)), 1
            )
        else:
            raise NotImplementedError(
                self.__class__.__name__ + ".blob shape not implemented"
            )

    def __y_shape(
        self, y: NDArray, t: NDArray, periodic_y: bool, Ly: float,
    ) -> NDArray:
        y_diffs = y - self.__get_y_blob_pos(t)
        if periodic_y:
            # shift of Ly/2 needed for modulo operator
            return (
                1
                / np.sqrt(np.pi)
                * np.exp(-(((y_diffs + Ly / 2) % Ly - Ly / 2) ** 2) / self.width_y ** 2)
            )
        else:
            return 1 / np.sqrt(np.pi) * np.exp(-(y_diffs ** 2) / self.width_y ** 2)

    def __get_x_blob_pos(self, t: NDArray) -> NDArray:
        return self.pos_x + self.v_x * (t - self.t_init)

    def __get_y_blob_pos(self, t: NDArray) -> NDArray:
        return self.pos_y + self.v_y * (t - self.t_init)
