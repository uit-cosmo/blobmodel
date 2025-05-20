import pytest
from blobmodel import AbstractBlobShape, BlobShapeImpl, BlobShapeEnum
import numpy as np


def test_gauss_pulse_shape():
    """
    Tests that gaussian pulse shape has expected shape.
    """
    ps = BlobShapeImpl()
    x = np.arange(-10, 10, 0.1)
    values = ps.get_blob_shape_perp(x)
    expected_values = 1 / np.sqrt(np.pi) * np.exp(-(x**2))
    assert np.max(np.abs(values - expected_values)) < 1e-5, "Wrong gaussian shape"


def test_kwargs():
    """
    Tests that additional shape parameters provided through kwargs are correctly implemented.
    """
    lam = 0.5
    x = np.arange(-10, 10, 0.1)
    expected_values = np.zeros(len(x))
    expected_values[x < 0] = np.exp(x[x < 0] / lam)
    expected_values[x >= 0] = np.exp(-x[x >= 0] / (1 - lam))

    ps = BlobShapeImpl(BlobShapeEnum.double_exp, BlobShapeEnum.double_exp)
    values = ps.get_blob_shape_perp(x, lam=0.5)
    assert np.max(np.abs(values - expected_values)) < 1e-5, "Wrong shape"


def test__get_double_exponential_shape():
    """
    Tests that double exponential has expected shape.
    """
    theta = np.array([-1, 0, 1])
    lam = 0.5
    expected_result = np.array([0.13533528, 1.0, 0.13533528])
    ps = BlobShapeImpl(BlobShapeEnum.double_exp, BlobShapeEnum.double_exp)
    values = ps.get_blob_shape_perp(theta, lam=lam)
    assert np.max(np.abs(values - expected_result)) < 1e-5, "Wrong shape"


def test__get_secant_shape():
    """
    Tests that secant has expected shape.
    """
    theta = np.array([1, 2, 3])
    expected_result = np.array([0.20628208, 0.08460748, 0.03161706])

    ps = BlobShapeImpl(BlobShapeEnum.secant, BlobShapeEnum.secant)
    values = ps.get_blob_shape_perp(theta)
    assert np.max(np.abs(values - expected_result)) < 1e-5, "Wrong shape"


def test__get_lorentz_shape():
    """
    Tests that lorentz has expected shape.
    """
    theta = np.array([1, 2, 3])
    expected_result = np.array([0.15915494, 0.06366198, 0.03183099])

    ps = BlobShapeImpl(BlobShapeEnum.lorentz, BlobShapeEnum.lorentz)
    values = ps.get_blob_shape_perp(theta)
    assert np.max(np.abs(values - expected_result)) < 1e-5, "Wrong shape"


def test__get_dipole_shape():
    """
    Tests that dipole has expected shape.
    """
    theta = np.array([1, 2, 3])
    expected_result = np.array([-0.48394145, -0.21596387, -0.02659109])

    ps = BlobShapeImpl(BlobShapeEnum.dipole, BlobShapeEnum.dipole)
    values = ps.get_blob_shape_perp(theta)
    assert np.max(np.abs(values - expected_result)) < 1e-5, "Wrong shape"
