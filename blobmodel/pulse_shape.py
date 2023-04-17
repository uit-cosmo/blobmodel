from abc import ABC, abstractmethod
import numpy as np


class AbstractBlobShape(ABC):
    """Abstract class containing the blob pulse shapes. Two-dimensional blob
    pulse shapes are written in the form.

    phi(theta_x, theta_y) = phi_x(theta_x) * phi_y(theta_y).
    """

    @abstractmethod
    def get_pulse_shape_prop(self, theta: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_pulse_shape_perp(self, theta: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError


class BlobShapeImpl(AbstractBlobShape):
    """Implementation of the AbstractPulseShape class."""

    def __init__(self, pulse_shape_prop="gauss", pulse_shape_perp="gauss"):
        if (
            pulse_shape_prop not in BlobShapeImpl.__GENERATORS__.keys()
            or pulse_shape_perp not in BlobShapeImpl.__GENERATORS__.keys()
        ):
            raise NotImplementedError(
                f"{self.__class__.__name__}.blob_shape not implemented"
            )
        self.get_pulse_shape_prop = BlobShapeImpl.__GENERATORS__.get(pulse_shape_prop)
        self.get_pulse_shape_perp = BlobShapeImpl.__GENERATORS__.get(pulse_shape_perp)

    def get_pulse_shape_prop(self, theta: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def get_pulse_shape_perp(self, theta: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _get_exponential_shape(theta: np.ndarray, **kwargs) -> np.ndarray:
        return np.exp(theta) * np.heaviside(-1.0 * theta, 1)

    @staticmethod
    def _get_lorentz_shape(theta: np.ndarray, **kwargs) -> np.ndarray:
        return 1 / (np.pi * (1 + theta**2))

    @staticmethod
    def _get_double_exponential_shape(theta: np.ndarray, **kwargs) -> np.ndarray:
        lam = kwargs["lam"]
        assert (lam > 0.0) & (lam < 1.0)
        kern = np.zeros(shape=np.shape(theta))
        kern[theta < 0] = np.exp(theta[theta < 0] / lam)
        kern[theta >= 0] = np.exp(-theta[theta >= 0] / (1 - lam))
        return kern

    @staticmethod
    def _get_gaussian_shape(theta: np.ndarray, **kwargs) -> np.ndarray:
        return 1 / np.sqrt(np.pi) * np.exp(-(theta**2))

    @staticmethod
    def _get_secant_shape(theta: np.ndarray, **kwargs) -> np.ndarray:
        return 2 / np.pi / (np.exp(theta) + np.exp(-theta))

    __GENERATORS__ = {
        "exp": _get_exponential_shape,
        "gauss": _get_gaussian_shape,
        "2-exp": _get_double_exponential_shape,
        "lorentz": _get_lorentz_shape,
        "secant": _get_secant_shape,
    }
