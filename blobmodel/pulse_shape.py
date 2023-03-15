from abc import ABC, abstractmethod
import numpy as np
from nptyping import NDArray


class AbstractBlobShape(ABC):
    """
    Abstract class containing the blob pulse shapes. Two-dimensional blob pulse shapes are written in the form
    phi(theta_x, theta_y) = phi_x(theta_x) * phi_y(theta_y).
    """

    @abstractmethod
    def pulse_shape_prop(self, theta_prop: NDArray):
        raise NotImplementedError

    @abstractmethod
    def pulse_shape_perp(self, theta_perp: NDArray):
        raise NotImplementedError


class BlobShapeImpl(AbstractBlobShape):
    """
    Implementation of the AbstractPulseShape class.
    """

    def __init__(self, pulse_shape):
        self.pulse_shape = pulse_shape
        if pulse_shape not in ["gauss", "exp"]:
            raise NotImplementedError(
                f"{self.__class__.__name__}.blob_shape not implemented"
            )

    def pulse_shape_prop(self, theta_prop: NDArray):
        if self.pulse_shape == "gauss":
            return 1 / np.sqrt(np.pi) * np.exp(-(theta_prop**2))
        elif self.pulse_shape == "exp":
            return np.exp(theta_prop) * np.heaviside(-1.0 * theta_prop, 1)

    def pulse_shape_perp(self, theta_perp: NDArray):
        return 1 / np.sqrt(np.pi) * np.exp(-(theta_perp**2))
