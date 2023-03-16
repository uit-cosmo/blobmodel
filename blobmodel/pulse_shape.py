from abc import ABC, abstractmethod
import numpy as np
from nptyping import NDArray
from typing import Callable


class AbstractBlobShape(ABC):
    """
    Abstract class containing the blob pulse shapes. Two-dimensional blob pulse shapes are written in the form
    phi(theta_x, theta_y) = phi_x(theta_x) * phi_y(theta_y).
    """

    @abstractmethod
    def get_pulse_shape_prop(self, theta_prop: NDArray, kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_pulse_shape_perp(self, theta_perp: NDArray, kwargs):
        raise NotImplementedError


class BlobShapeImpl(AbstractBlobShape):
    """
    Implementation of the AbstractPulseShape class.
    """

    __SHAPE_NAMES__ = {"exp", "gauss", "2-exp", "lorentz", "secant"}

    def __init__(self, pulse_shape_prop="gauss", pulse_shape_perp="gauss"):
        self.pulse_shape_prop = pulse_shape_prop
        self.pulse_shape_perp = pulse_shape_perp
        if (
            pulse_shape_prop not in BlobShapeImpl.__SHAPE_NAMES__
            or pulse_shape_perp not in BlobShapeImpl.__SHAPE_NAMES__
        ):
            raise NotImplementedError(
                f"{self.__class__.__name__}.blob_shape not implemented"
            )

    def get_pulse_shape_prop(self, theta: np.ndarray, kwargs) -> np.ndarray:
        return self._get_generator(self.pulse_shape_prop)(theta, kwargs)

    def get_pulse_shape_perp(self, theta: np.ndarray, kwargs) -> np.ndarray:
        return self._get_generator(self.pulse_shape_perp)(theta, kwargs)

    @staticmethod
    def _get_generator(
        shape_name: str,
    ) -> Callable[[np.ndarray, dict], np.ndarray]:
        if shape_name == "exp":
            return BlobShapeImpl._get_exponential_shape
        if shape_name == "2-exp":
            return BlobShapeImpl._get_double_exponential_shape
        if shape_name == "lorentz":
            return BlobShapeImpl._get_lorentz_shape
        if shape_name == "gauss":
            return BlobShapeImpl._get_gaussian_shape
        if shape_name == "secant":
            return BlobShapeImpl._get_secant_shape

    @staticmethod
    def _get_exponential_shape(theta: np.ndarray, kwargs) -> np.ndarray:
        return np.exp(theta) * np.heaviside(-1.0 * theta, 1)

    @staticmethod
    def _get_lorentz_shape(theta: np.ndarray, kwargs) -> np.ndarray:
        return 1 / (np.pi * (1 + theta**2))

    @staticmethod
    def _get_double_exponential_shape(theta: np.ndarray, kwargs) -> np.ndarray:
        lam = kwargs["lam"]
        assert (lam > 0.0) & (lam < 1.0)
        kern = np.zeros(len(theta))
        kern[theta < 0] = np.exp(theta[theta < 0] / lam)
        kern[theta >= 0] = np.exp(-theta[theta >= 0] / (1 - lam))
        return kern

    @staticmethod
    def _get_gaussian_shape(theta: np.ndarray, kwargs) -> np.ndarray:
        return 1 / np.sqrt(np.pi) * np.exp(-(theta**2))

    @staticmethod
    def _get_secant_shape(theta: np.ndarray, kwargs) -> np.ndarray:
        return 2 / np.pi / (np.exp(theta) + np.exp(-theta))
