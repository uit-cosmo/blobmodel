import warnings
from typing import Tuple, Union
from nptyping import NDArray
import numpy as np


class Blob:
    """A single blob."""

    def __init__(
        self,
        blob_id: int,
        blob_shape: str,
        amplitude: float,
        width_prop: float,
        width_perp: float,
        v_x: float,
        v_y: float,
        pos_x: float,
        pos_y: float,
        t_init: float,
        t_drain: Union[float, NDArray],
    ) -> None:
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
        if self.v_x != 0:
            self.theta = np.arctan(self.v_y / self.v_x)
        else:
            self.theta = np.pi / 2 * np.sign(self.v_y)

    def discretize_blob(
        self,
        x: NDArray,
        y: NDArray,
        t: NDArray,
        Ly: float,
        periodic_y: bool = False,
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
        if (self.width_perp > 0.1 * Ly or self.width_prop > 0.1 * Ly) and periodic_y:
            warnings.warn("blob width big compared to Ly")

        x_perp, y_perp = self._rotate(
            origin=(self.pos_x, self.pos_y), x=x, y=y, angle=-self.theta
        )
        if not periodic_y:
            return self._single_blob(x_perp, y_perp, t, Ly, periodic_y)
        if np.sin(self.theta) == 0:
            _x_border = Ly - self.pos_y
            _adjusted_Ly = Ly
        else:
            _x_border = (Ly - self.pos_y) / np.sin(self.theta)
            _adjusted_Ly = Ly / np.sin(self.theta)
        if type(t) in [int, float]:
            # t has dimensionality = 0, used for testing
            _number_of_y_propagations = (
                self._prop_dir_blob_position(t) + _adjusted_Ly - _x_border
            ) // _adjusted_Ly
        else:
            _number_of_y_propagations = (
                self._prop_dir_blob_position(t)[0, 0] + _adjusted_Ly - _x_border
            ) // _adjusted_Ly
        return (
            self._single_blob(
                x_perp, y_perp, t, Ly, periodic_y, _number_of_y_propagations
            )
            + self._single_blob(
                x_perp,
                y_perp,
                t,
                Ly,
                periodic_y,
                _number_of_y_propagations,
                x_offset=Ly * np.sin(self.theta),
                y_offset=Ly * np.cos(self.theta),
            )
            + self._single_blob(
                x_perp,
                y_perp,
                t,
                Ly,
                periodic_y,
                _number_of_y_propagations,
                x_offset=-Ly * np.sin(self.theta),
                y_offset=-Ly * np.cos(self.theta),
            )
        )

    def _single_blob(
        self,
        x_perp: NDArray,
        y_perp: NDArray,
        t: NDArray,
        Ly: float,
        periodic_y: bool,
        number_of_y_propagations: NDArray = 0,
        x_offset: NDArray = 0,
        y_offset: NDArray = 0,
    ) -> NDArray:
        return (
            self.amplitude
            * self._drain(t)
            * self._propagation_direction_shape(
                x_perp + x_offset,
                t,
                Ly,
                periodic_y,
                number_of_y_propagations=number_of_y_propagations,
            )
            * self._perpendicular_direction_shape(
                y_perp + y_offset,
                Ly,
                periodic_y,
                number_of_y_propagations=number_of_y_propagations,
            )
            * self._blob_arrival(t)
        )

    def _drain(self, t: NDArray) -> NDArray:
        if isinstance(self.t_drain, (int, float)):
            return np.exp(-(t - self.t_init) / self.t_drain)
        return np.exp(-(t - self.t_init) / self.t_drain[np.newaxis, :, np.newaxis])

    def _blob_arrival(self, t: NDArray) -> NDArray:
        return np.heaviside(t - self.t_init, 1)

    def _propagation_direction_shape(
        self,
        x: NDArray,
        t: NDArray,
        Ly: float,
        periodic_y: bool,
        number_of_y_propagations: NDArray,
    ) -> NDArray:
        if periodic_y:
            x_diffs = (
                x
                - self._prop_dir_blob_position(t)
                + number_of_y_propagations * Ly * np.sin(self.theta)
            )
        else:
            x_diffs = x - self._prop_dir_blob_position(t)

        if self.blob_shape == "gauss":
            return 1 / np.sqrt(np.pi) * np.exp(-(x_diffs**2 / self.width_prop**2))
        elif self.blob_shape == "exp":
            return np.exp(x_diffs) * np.heaviside(-1.0 * (x_diffs), 1)
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__}.blob_shape not implemented"
            )

    def _perpendicular_direction_shape(
        self,
        y: NDArray,
        Ly: float,
        periodic_y: bool,
        number_of_y_propagations: NDArray,
    ) -> NDArray:
        if periodic_y:
            y_diffs = (
                y
                - self._perp_dir_blob_position()
                + number_of_y_propagations * Ly * np.cos(self.theta)
            )
        else:
            y_diffs = y - self._perp_dir_blob_position()
        return 1 / np.sqrt(np.pi) * np.exp(-(y_diffs**2) / self.width_perp**2)

    def _prop_dir_blob_position(self, t: NDArray) -> NDArray:
        return self.pos_x + (self.v_x**2 + self.v_y**2) ** 0.5 * (t - self.t_init)

    def _perp_dir_blob_position(self) -> NDArray:
        return self.pos_y

    def _rotate(
        self, origin: Tuple[float, float], x: NDArray, y: NDArray, angle: float
    ) -> Tuple[float, float]:
        ox, oy = origin
        px, py = x, y

        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy
