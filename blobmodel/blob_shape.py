"""This module defines classes for blob pulse shapes used in two-dimensional simulations."""

from enum import Enum
from abc import ABC, abstractmethod
import numpy as np


class BlobShapeEnum(Enum):
    """
    Enum class representing blob shapes.
    """

    exp = 1
    lorentz = 2
    double_exp = 3
    gaussian = 4
    rect = 5
    secant = 6
    dipole = 7


class AbstractBlobShape(ABC):
    """Abstract class containing the blob pulse shapes. Two-dimensional blob
    pulse shapes are written in the form:

    ``phi(theta_p, theta_s) = phi_p(theta_p) * phi_s(theta_s)``

    Where the p and s subindexes stand for primary and secondary directions.
    """

    @abstractmethod
    def get_blob_shape_p(self, theta: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_blob_shape_s(self, theta: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError


def _get_exponential_shape(theta: np.ndarray, **kwargs) -> np.ndarray:
    """Compute the exponential pulse shape.

    Parameters
    ----------
    theta : np.ndarray
        Array of theta values.
    kwargs
        Additional keyword arguments.

    Returns
    -------
    np.ndarray
        Array representing the exponential pulse shape.
    """
    kern = np.zeros(shape=np.shape(theta))
    kern[theta < 0] = np.exp(theta[theta < 0])
    return kern


def _get_lorentz_shape(theta: np.ndarray, **kwargs) -> np.ndarray:
    """Compute the Lorentzian pulse shape.

    Parameters
    ----------
    theta : np.ndarray
        Array of theta values.
    kwargs
        Additional keyword arguments.

    Returns
    -------
    np.ndarray
        Array representing the Lorentzian pulse shape.
    """
    return 1 / (np.pi * (1 + theta**2))


def _get_double_exponential_shape(theta: np.ndarray, **kwargs) -> np.ndarray:
    """Compute the double-exponential pulse shape.

    The shape is ``exp(-theta / lam)`` for ``theta >= 0`` and
    ``exp(theta / (1 - lam))`` for ``theta < 0``, i.e. ``lam`` is the
    e-folding fraction of the leading (``theta >= 0``) side of the pulse.
    For a blob propagating with ``v_x > 0`` observed at a fixed position,
    ``lam`` is therefore the temporal *rise* fraction of the measured pulse,
    matching the asymmetry-parameter convention of the FPP literature.

    .. versionchanged:: 2.0.0
        The convention was flipped: ``lam`` previously weighted the trailing
        (``theta < 0``) side, so ``lam`` was the temporal *fall* fraction.
        Code written against older versions that passed ``1 - lam`` to
        compensate should now pass ``lam`` directly.

    Parameters
    ----------
    theta : np.ndarray
        Array of theta values.
    kwargs
        Additional keyword arguments.
        lam : float
            Asymmetry parameter controlling the shape, in the interval [0, 1].
            The limits are the one-sided shapes: lam = 0 is nonzero only for
            theta < 0 (a pure temporal decay for v_x > 0), lam = 1 is nonzero
            only for theta >= 0 (a pure temporal rise for v_x > 0).

    Returns
    -------
    np.ndarray
        Array representing the double-exponential pulse shape.

    Raises
    ------
    ValueError
        If lam lies outside the interval [0, 1].
    """
    lam = kwargs["lam"]
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in the interval [0, 1], got lam = {lam}.")
    kern = np.zeros(shape=np.shape(theta))
    if lam < 1.0:
        kern[theta < 0] = np.exp(theta[theta < 0] / (1 - lam))
    if lam > 0.0:
        kern[theta >= 0] = np.exp(-theta[theta >= 0] / lam)
    return kern


def _get_gaussian_shape(theta: np.ndarray, **kwargs) -> np.ndarray:
    """Compute the Gaussian pulse shape.

    Parameters
    ----------
    theta : np.ndarray
        Array of theta values.
    kwargs
        Additional keyword arguments.

    Returns
    -------
    np.ndarray
        Array representing the Gaussian pulse shape.
    """
    return 1 / np.sqrt(np.pi) * np.exp(-(theta**2))


def _get_rectangle_shape(theta: np.ndarray, **kwargs) -> np.ndarray:
    """Compute the hard ellipse pulse shape.
    Parameters
    ----------
    theta : np.ndarray
        Array of theta values.
    kwargs
        Additional keyword arguments.
    Returns
    -------
    np.ndarray
        Array representing the rectangle pulse shape.
    """
    return np.abs(theta) < 0.5


def _get_secant_shape(theta: np.ndarray, **kwargs) -> np.ndarray:
    """Compute the secant pulse shape.

    Parameters
    ----------
    theta : np.ndarray
        Array of theta values.
    kwargs
        Additional keyword arguments.

    Returns
    -------
    np.ndarray
        Array representing the secant pulse shape.
    """
    return 2 / np.pi / (np.exp(theta) + np.exp(-theta))


def _get_dipole_shape(theta: np.ndarray, **kwargs) -> np.ndarray:
    """Compute the diople pulse shape as a derivative of a gaussian pulse shape.

    Parameters
    ----------
    theta : np.ndarray
        Array of theta values.
    kwargs
        Additional keyword arguments.

    Returns
    -------
    np.ndarray
        Array representing the dipole pulse shape.
    """
    return -2 * theta / np.sqrt(2 * np.pi) * np.exp(-(theta**2) / 2)


class BlobShapeImpl(AbstractBlobShape):
    """Implementation of the AbstractBlobShape class."""

    def __init__(
        self,
        pulse_shape_p: BlobShapeEnum = BlobShapeEnum.gaussian,
        pulse_shape_s: BlobShapeEnum = BlobShapeEnum.gaussian,
    ):
        """Initialize the BlobShapeImpl object.

        Parameters
        ----------
        pulse_shape_p : BlobShapeEnum, optional
            Type of pulse shape in the principal direction,
            by default BlobShapeEnum.gaussian.
        pulse_shape_s : BlobShapeEnum, optional
            Type of pulse shape in the secondary direction,
            by default BlobShapeEnum.gaussian.

        Raises
        ------
        NotImplementedError
            If a pulse shape is not a member of BlobShapeEnum with an
            implemented shape function.
        """
        if (
            pulse_shape_p not in BlobShapeImpl.__GENERATORS
            or pulse_shape_s not in BlobShapeImpl.__GENERATORS
        ):
            raise NotImplementedError(
                f"{self.__class__.__name__}.blob_shape not implemented"
            )
        self._shape_p = BlobShapeImpl.__GENERATORS[pulse_shape_p]
        self._shape_s = BlobShapeImpl.__GENERATORS[pulse_shape_s]

    def get_blob_shape_p(self, theta: np.ndarray, **kwargs) -> np.ndarray:
        """Compute the pulse shape in the principal direction.

        Parameters
        ----------
        theta : np.ndarray
            Array of theta values.
        kwargs
            Additional keyword arguments passed to the shape function.

        Returns
        -------
        np.ndarray
            Array representing the pulse shape in the principal direction.
        """
        return self._shape_p(theta, **kwargs)

    def get_blob_shape_s(self, theta: np.ndarray, **kwargs) -> np.ndarray:
        """Compute the pulse shape in the secondary direction.

        Parameters
        ----------
        theta : np.ndarray
            Array of theta values.
        kwargs
            Additional keyword arguments passed to the shape function.

        Returns
        -------
        np.ndarray
            Array representing the pulse shape in the secondary direction.
        """
        return self._shape_s(theta, **kwargs)

    __GENERATORS = {
        BlobShapeEnum.exp: _get_exponential_shape,
        BlobShapeEnum.lorentz: _get_lorentz_shape,
        BlobShapeEnum.double_exp: _get_double_exponential_shape,
        BlobShapeEnum.gaussian: _get_gaussian_shape,
        BlobShapeEnum.secant: _get_secant_shape,
        BlobShapeEnum.dipole: _get_dipole_shape,
        BlobShapeEnum.rect: _get_rectangle_shape,
    }
