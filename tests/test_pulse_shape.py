import pytest
from blobmodel import BlobShapeImpl
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
