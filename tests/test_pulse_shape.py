import pytest
from blobmodel import BlobShapeImpl, AbstractBlobShape, BlobShapeImpl
import numpy as np


def test_gauss_pulse_shape():
    ps = BlobShapeImpl()
    x = np.arange(-10, 10, 0.1)
    values = ps.get_blob_shape_perp(x)
    expected_values = 1 / np.sqrt(np.pi) * np.exp(-(x**2))
    assert np.max(np.abs(values - expected_values)) < 1e-5, "Wrong gaussian shape"


def test_throw_unknown_shape():
    with pytest.raises(NotImplementedError):
        BlobShapeImpl("LOL")


def test_kwargs():
    lam = 0.5
    x = np.arange(-10, 10, 0.1)
    expected_values = np.zeros(len(x))
    expected_values[x < 0] = np.exp(x[x < 0] / lam)
    expected_values[x >= 0] = np.exp(-x[x >= 0] / (1 - lam))

    ps = BlobShapeImpl("2-exp", "2-exp")
    values = ps.get_blob_shape_perp(x, lam=0.5)
    assert np.max(np.abs(values - expected_values)) < 1e-5, "Wrong shape"


def test_abstract_mehtods():
    AbstractBlobShape.__abstractmethods__ = set()

    class MyShape(AbstractBlobShape):
        pass

    my_obj = MyShape()

    with pytest.raises(NotImplementedError):
        my_obj.get_blob_shape_prop([0, 1, 2])

    with pytest.raises(NotImplementedError):
        my_obj.get_blob_shape_perp([0, 1, 2])


def test__get_double_exponential_shape():
    theta = np.array([-1, 0, 1])
    lam = 0.5
    expected_result = np.array([0.13533528, 1.0, 0.13533528])
    assert np.allclose(
        BlobShapeImpl._get_double_exponential_shape(theta, lam=lam), expected_result
    )


def test__get_secant_shape():
    theta = np.array([1, 2, 3])
    expected_result = np.array([0.20628208, 0.08460748, 0.03161706])
    assert np.allclose(BlobShapeImpl._get_secant_shape(theta), expected_result)


def test__get_lorentz_shape():
    theta = np.array([1, 2, 3])
    expected_result = np.array([0.15915494, 0.06366198, 0.03183099])
    assert np.allclose(BlobShapeImpl._get_lorentz_shape(theta), expected_result)
