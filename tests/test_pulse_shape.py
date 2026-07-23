import pytest
from blobmodel import AbstractBlobShape, Blob, BlobShapeImpl, BlobShapeEnum
import numpy as np


def test_double_exp_lam_weights_leading_side():
    """
    Locks the double-exponential asymmetry convention: lam is the e-folding
    fraction of the leading (theta >= 0) side, so the shape is
    exp(-theta / lam) for theta >= 0 and exp(theta / (1 - lam)) for
    theta < 0 (the FPP-literature convention; flipped in 2.0.0).
    """
    lam = 0.2
    theta = np.arange(-10, 10, 0.1)
    expected_values = np.zeros(len(theta))
    expected_values[theta < 0] = np.exp(theta[theta < 0] / (1 - lam))
    expected_values[theta >= 0] = np.exp(-theta[theta >= 0] / lam)

    ps = BlobShapeImpl(BlobShapeEnum.double_exp, BlobShapeEnum.double_exp)
    values = ps.get_blob_shape_p(theta, lam=lam)
    np.testing.assert_allclose(values, expected_values)


def test_double_exp_lam_is_temporal_rise_fraction():
    """
    For a blob propagating with v_x > 0 observed at a fixed position, lam is
    the e-folding time fraction of the temporal *rise* of the measured pulse
    and 1 - lam that of the fall.
    """
    lam = 0.2
    blob = Blob(
        blob_shape=BlobShapeImpl(BlobShapeEnum.double_exp, BlobShapeEnum.gaussian),
        shape_parameters_p={"lam": lam},
    )
    x_probe = 10.0
    t = np.linspace(0, 20, 501)
    signal = blob.discretize_blob(
        x=np.array(x_probe)[np.newaxis, np.newaxis, np.newaxis],
        y=np.array(0.0)[np.newaxis, np.newaxis, np.newaxis],
        t=t[np.newaxis, np.newaxis, :],
        Ly=0,
        one_dimensional=True,
    )[0, 0, :]
    # Default blob: amplitude 1, width_p 1, v_x 1, t_init 0, so the blob
    # center passes the probe at t = x_probe. Before that the signal rises
    # with time scale lam, after it falls with time scale 1 - lam.
    expected = np.where(
        t <= x_probe,
        np.exp(-(x_probe - t) / lam),
        np.exp(-(t - x_probe) / (1 - lam)),
    )
    np.testing.assert_allclose(signal, expected)


def test_gauss_pulse_shape():
    """
    Tests that gaussian pulse shape has expected shape.
    """
    ps = BlobShapeImpl()
    x = np.arange(-10, 10, 0.1)
    values = ps.get_blob_shape_s(x)
    expected_values = 1 / np.sqrt(np.pi) * np.exp(-(x**2))
    assert np.max(np.abs(values - expected_values)) < 1e-5, "Wrong gaussian shape"


def test_kwargs():
    """
    Tests that additional shape parameters provided through kwargs are correctly implemented.
    """
    lam = 0.5
    x = np.arange(-10, 10, 0.1)
    expected_values = np.zeros(len(x))
    expected_values[x < 0] = np.exp(x[x < 0] / (1 - lam))
    expected_values[x >= 0] = np.exp(-x[x >= 0] / lam)

    ps = BlobShapeImpl(BlobShapeEnum.double_exp, BlobShapeEnum.double_exp)
    values = ps.get_blob_shape_s(x, lam=0.5)
    assert np.max(np.abs(values - expected_values)) < 1e-5, "Wrong shape"


def test__get_double_exponential_shape():
    """
    Tests that double exponential has expected shape.
    """
    theta = np.array([-1, 0, 1])
    lam = 0.5
    expected_result = np.array([0.13533528, 1.0, 0.13533528])
    ps = BlobShapeImpl(BlobShapeEnum.double_exp, BlobShapeEnum.double_exp)
    values = ps.get_blob_shape_s(theta, lam=lam)
    assert np.max(np.abs(values - expected_result)) < 1e-5, "Wrong shape"


def test__get_secant_shape():
    """
    Tests that secant has expected shape.
    """
    theta = np.array([1, 2, 3])
    expected_result = np.array([0.20628208, 0.08460748, 0.03161706])

    ps = BlobShapeImpl(BlobShapeEnum.secant, BlobShapeEnum.secant)
    values = ps.get_blob_shape_s(theta)
    assert np.max(np.abs(values - expected_result)) < 1e-5, "Wrong shape"


def test__get_lorentz_shape():
    """
    Tests that lorentz has expected shape.
    """
    theta = np.array([1, 2, 3])
    expected_result = np.array([0.15915494, 0.06366198, 0.03183099])

    ps = BlobShapeImpl(BlobShapeEnum.lorentz, BlobShapeEnum.lorentz)
    values = ps.get_blob_shape_s(theta)
    assert np.max(np.abs(values - expected_result)) < 1e-5, "Wrong shape"


def test__get_dipole_shape():
    """
    Tests that dipole has expected shape.
    """
    theta = np.array([1, 2, 3])
    expected_result = np.array([-0.48394145, -0.21596387, -0.02659109])

    ps = BlobShapeImpl(BlobShapeEnum.dipole, BlobShapeEnum.dipole)
    values = ps.get_blob_shape_s(theta)
    assert np.max(np.abs(values - expected_result)) < 1e-5, "Wrong shape"
