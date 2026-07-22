"""Tests for input validation: Blob, Model, BlobFactory and pulse shape
arguments raise typed errors instead of asserts (feedback items 6, 7, 8, 10)."""

import warnings

import numpy as np
import pytest

from blobmodel import (
    Geometry,
    Blob,
    BlobShapeEnum,
    BlobShapeImpl,
    DefaultBlobFactory,
    DistributionEnum,
    Model,
)


def make_blob(**kwargs):
    """Return a valid Blob, with any parameter overridable through kwargs."""
    parameters = dict(
        blob_id=0,
        blob_shape=BlobShapeImpl(),
        amplitude=1,
        width_p=1,
        width_s=1,
        v_x=1,
        v_y=1,
        pos_x0=0,
        pos_y0=0,
        t_init=0,
        t_drain=10,
    )
    parameters.update(kwargs)
    return Blob(**parameters)


def test_blob_rejects_nonpositive_widths():
    """
    Blob widths must be positive; zero or negative values raise ValueError.
    """
    for bad_widths in [
        {"width_p": 0},
        {"width_p": -1},
        {"width_s": 0},
        {"width_s": -0.5},
    ]:
        with pytest.raises(ValueError, match="width"):
            make_blob(**bad_widths)


def test_blob_rejects_nonpositive_t_drain():
    """
    t_drain is a drain time scale and must be positive, both as a scalar and
    element-wise as an array.
    """
    for bad_t_drain in [0, -1, np.array([1.0, -1.0])]:
        with pytest.raises(ValueError, match="t_drain"):
            make_blob(t_drain=bad_t_drain)


def test_blob_accepts_positive_t_drain():
    """
    Positive scalars (including inf) and all-positive arrays are valid t_drain
    values.
    """
    make_blob(t_drain=np.inf)
    make_blob(t_drain=np.array([1.0, 2.0]))


def test_blob_rejects_wrong_blob_shape_type():
    """
    blob_shape must be an AbstractBlobShape instance.
    """
    with pytest.raises(TypeError, match="blob_shape"):
        make_blob(blob_shape="gauss")


def test_discretize_blob_one_dimensional_requires_ly_zero():
    """
    Calling discretize_blob with one_dimensional=True and Ly != 0 raises
    ValueError instead of an assert.
    """
    blob = make_blob()
    x = np.arange(0, 10, 0.1)
    y = np.array([0.0])
    mesh_x, mesh_y = np.meshgrid(x, y)
    with pytest.raises(ValueError, match="Ly"):
        blob.discretize_blob(x=mesh_x, y=mesh_y, t=0, Ly=10, one_dimensional=True)


def test_discretize_blob_accepts_numpy_scalar_time():
    """
    discretize_blob with periodic_y accepts numpy scalar times, which the old
    `type(t) in [int, float]` check misclassified as arrays.
    """
    blob = make_blob(v_y=5)
    x = np.arange(0, 10, 0.1)
    y = np.array([0, 1])
    mesh_x, mesh_y = np.meshgrid(x, y)

    reference = blob.discretize_blob(x=mesh_x, y=mesh_y, t=1.0, periodic_y=True, Ly=10)
    values = blob.discretize_blob(
        x=mesh_x, y=mesh_y, t=np.float64(1.0), periodic_y=True, Ly=10
    )
    np.testing.assert_array_equal(values, reference)


def test_double_exp_lam_outside_unit_interval_raises():
    """
    The double-exponential asymmetry parameter lam must lie in [0, 1].
    """
    ps = BlobShapeImpl(BlobShapeEnum.double_exp, BlobShapeEnum.double_exp)
    theta = np.linspace(-5, 5, 100)
    for bad_lam in [-0.1, 1.1]:
        with pytest.raises(ValueError, match="lam"):
            ps.get_blob_shape_p(theta, lam=bad_lam)


def test_double_exp_lam_limits_are_one_sided():
    """
    lam = 0 and lam = 1 are valid one-sided limits of the double-exponential
    shape: a pure decay and a pure rise, respectively.
    """
    ps = BlobShapeImpl(BlobShapeEnum.double_exp, BlobShapeEnum.double_exp)
    theta = np.linspace(-5, 5, 100)

    pure_decay = ps.get_blob_shape_p(theta, lam=0)
    expected_decay = np.zeros_like(theta)
    expected_decay[theta >= 0] = np.exp(-theta[theta >= 0])
    np.testing.assert_allclose(pure_decay, expected_decay)

    pure_rise = ps.get_blob_shape_p(theta, lam=1)
    expected_rise = np.zeros_like(theta)
    expected_rise[theta < 0] = np.exp(theta[theta < 0])
    np.testing.assert_allclose(pure_rise, expected_rise)


def test_model_rejects_wrong_types():
    """
    Model raises TypeError for blob_shape/blob_factory of the wrong type.
    """
    with pytest.raises(TypeError, match="blob_shape"):
        Model(blob_shape="gauss")
    with pytest.raises(TypeError, match="blob_factory"):
        Model(blob_factory="default")


def test_model_rejects_bad_t_drain():
    """
    Model raises ValueError for t_drain arrays of the wrong length and for
    non-positive t_drain values.
    """
    with pytest.raises(ValueError, match="t_drain"):
        Model(
            geometry=Geometry(Nx=5),
            t_drain=np.ones(3),
        )
    with pytest.raises(ValueError, match="t_drain"):
        Model(t_drain=-1)
    with pytest.raises(ValueError, match="t_drain"):
        Model(
            geometry=Geometry(Nx=2),
            t_drain=np.array([1.0, 0.0]),
        )


def test_periodic_y_width_warning_fired_once_with_values():
    """
    The blob-width warning for periodic_y is emitted once per realization from
    Model (not once per blob) and reports the offending width and Ly.
    """
    model = Model(
        geometry=Geometry(Nx=5, Ny=5, Lx=5, Ly=1, dt=1, T=2, periodic_y=True),
        num_blobs=10,
        verbose=False,
        seed=42,
    )
    with pytest.warns(UserWarning, match="mirrored blobs") as record:
        model.make_realization()

    width_warnings = [w for w in record if "mirrored blobs" in str(w.message)]
    assert len(width_warnings) == 1
    assert "Ly = 1" in str(width_warnings[0].message)


def test_periodic_y_no_width_warning_for_small_widths():
    """
    No width warning is emitted when the blob widths are small compared to Ly.
    """
    model = Model(
        geometry=Geometry(Nx=5, Ny=5, Lx=5, Ly=10, dt=1, T=2, periodic_y=True),
        num_blobs=10,
        verbose=False,
        seed=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.make_realization()


def test_blob_junk_attribute_removed():
    """
    Blob no longer stores the int builtin as an attribute.
    """
    assert not hasattr(make_blob(), "int")
