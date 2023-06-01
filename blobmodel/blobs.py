import warnings
from typing import Tuple, Union
from nptyping import NDArray
import numpy as np
from .pulse_shape import AbstractBlobShape


class Blob:
    """A single blob."""

    def __init__(
        self,
        blob_id: int,
        blob_shape: AbstractBlobShape,
        amplitude: float,
        width_prop: float,
        width_perp: float,
        velocity_x: float,
        velocity_y: float,
        pos_x: float,
        pos_y: float,
        t_init: float,
        t_drain: Union[float, NDArray],
        prop_shape_parameters: dict = None,
        perp_shape_parameters: dict = None,
        blob_alignment: bool = True,
    ) -> None:
        self.int = int
        self.blob_id = blob_id
        self.blob_shape = blob_shape
        self.amplitude = amplitude
        self.width_prop = width_prop
        self.width_perp = width_perp
        self.v_x = velocity_x
        self.v_y = velocity_y
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
        self.theta = 0
        self.blob_alignment = blob_alignment
        if blob_alignment:
            import cmath

            self.theta = cmath.phase(self.v_x + self.v_y * 1j)

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
        The following blob shapes are implemented:
                gauss: 2D gaussian function
                exp: one sided exponential in x and gaussian in y

                Returns
                -------
                discretized blob on 3d array with dimensions x,y and t : np.array
        """
        # If one_dimensional, then Ly should be 0.
        assert (one_dimensional and Ly == 0) or not one_dimensional

        if (self.width_perp > 0.1 * Ly or self.width_prop > 0.1 * Ly) and periodic_y:
            warnings.warn("blob width big compared to Ly")

        x_perp, y_perp = self._rotate(
            origin=(self.pos_x, self.pos_y), x=x, y=y, angle=-self.theta
        )
        if not periodic_y or one_dimensional:
            return self._single_blob(
                x_perp, y_perp, t, Ly, periodic_y, one_dimensional=one_dimensional
            )
        if self.theta == 0:
            number_of_y_propagations = 0
        else:
            x_border = (Ly - self.pos_y) / np.sin(self.theta)
            adjusted_Ly = Ly / np.sin(self.theta)
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
                x_perp, y_perp, t, Ly, periodic_y, number_of_y_propagations
            )
            + self._single_blob(
                x_perp,
                y_perp,
                t,
                Ly,
                periodic_y,
                number_of_y_propagations,
                x_offset=Ly * np.sin(self.theta),
                y_offset=Ly * np.cos(self.theta),
                one_dimensional=one_dimensional,
            )
            + self._single_blob(
                x_perp,
                y_perp,
                t,
                Ly,
                periodic_y,
                number_of_y_propagations,
                x_offset=-Ly * np.sin(self.theta),
                y_offset=-Ly * np.cos(self.theta),
                one_dimensional=one_dimensional,
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
        one_dimensional: bool = False,
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

    def _drain(self, t: NDArray) -> NDArray:
        if isinstance(self.t_drain, (int, float)):
            return np.exp(-(t - self.t_init) / float(self.t_drain))
        return np.exp(-(t - self.t_init) / self.t_drain[np.newaxis, :, np.newaxis])

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
        theta_x = x_diffs / self.width_prop
        return self.blob_shape.get_pulse_shape_prop(
            theta_x, **self.prop_shape_parameters
        )

    def _perpendicular_direction_shape(
        self,
        y: NDArray,
        t: NDArray,
        Ly: float,
        periodic_y: bool,
        number_of_y_propagations: NDArray,
    ) -> NDArray:
        if periodic_y:
            y_diffs = (
                y
                - self._perp_dir_blob_position(t)
                + number_of_y_propagations * Ly * np.cos(self.theta)
            )
        else:
            y_diffs = y - self._perp_dir_blob_position(t)
        theta_y = y_diffs / self.width_perp
        return self.blob_shape.get_pulse_shape_perp(
            theta_y, **self.perp_shape_parameters
        )

    def _prop_dir_blob_position(self, t: NDArray) -> NDArray:
        return (
            self.pos_x + (self.v_x**2 + self.v_y**2) ** 0.5 * (t - self.t_init)
            if self.blob_alignment
            else self.pos_x + self.v_x * (t - self.t_init)
        )

    def _perp_dir_blob_position(self, t: NDArray) -> float:
        return (
            self.pos_y
            if self.blob_alignment
            else self.pos_y + self.v_y * (t - self.t_init)
        )

    def _rotate(
        self, origin: Tuple[float, float], x: NDArray, y: NDArray, angle: float
    ) -> Tuple[NDArray, NDArray]:
        ox, oy = origin
        px, py = x, y

        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy
