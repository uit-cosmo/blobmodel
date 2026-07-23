import pytest
from blobmodel import (
    Geometry,
    Model,
    DefaultBlobFactory,
    DistributionEnum,
    BlobShapeImpl,
    BlobShapeEnum,
)
import numpy as np

# use DefaultBlobFactory to define distribution functions fo random variables
bf = DefaultBlobFactory(
    A_dist=DistributionEnum.deg, vy_dist=DistributionEnum.zeros, t_drain=2
)

one_dim_model = Model(
    geometry=Geometry(Nx=100, Ny=1, Lx=10, Ly=0, dt=1, T=1000, periodic_y=False),
    blob_shape=BlobShapeImpl(BlobShapeEnum.exp, BlobShapeEnum.gaussian),
    num_blobs=10000,
    blob_factory=bf,
    one_dimensional=True,
)


def test_one_dim_converges_to_analytical():
    """
    Checks that one-dimensional realizations of the process agree with analytical results (see
    O. E. Garcia et al. Phys. Plasmas 1 May 2016; 23 (5): 052308. https://doi.org/10.1063/1.4951016)
    """
    ds = one_dim_model.make_realization(truncation_error=1e-2)
    model_profile = ds.n.mean(dim=("t"))

    x = np.linspace(0, 10, 100)
    t_p = 1
    t_w = 1 / 10
    amp = 1
    v_p = 1.0
    t_loss = 2.0
    t_d = t_loss * t_p / (t_loss + t_p)

    analytical_profile = t_d / t_w * amp * np.exp(-x / (v_p * t_loss))

    error = np.mean(abs(model_profile.values - analytical_profile))

    assert error < 0.1, "Numerical error too big"


def test_1d_geometry_mismatch_raises():
    """
    A one-dimensional model rejects a geometry that is not Ny=1, Ly=0
    (the geometry is user-provided, so it is not silently overwritten).
    """
    with pytest.raises(ValueError, match="Ny=1"):
        Model(
            geometry=Geometry(
                Nx=100, Ny=100, Lx=10, Ly=10, dt=1, T=1000, periodic_y=False
            ),
            blob_shape=BlobShapeImpl(BlobShapeEnum.exp, BlobShapeEnum.gaussian),
            num_blobs=1,
            blob_factory=bf,
            one_dimensional=True,
        )


def test_one_dim_dataset_has_no_y_dim():
    """
    One-dimensional output (Ly = 0) is returned already squeezed: n(x, t)
    without a y dimension, and blob_labels follows the same shape.
    """
    model = Model(
        geometry=Geometry(Nx=10, Ny=1, Lx=10, Ly=0, dt=1, T=10, periodic_y=False),
        num_blobs=2,
        blob_factory=bf,
        one_dimensional=True,
        labels="individual",
        verbose=False,
    )
    ds = model.make_realization()
    assert ds.n.dims == ("x", "t")
    assert "y" not in ds.dims
    assert ds.blob_labels.dims == ("x", "t")


def test_1d_default_geometry():
    """
    With no geometry given, a one-dimensional model builds a default
    Ny=1, Ly=0 geometry instead of the 2D default.
    """
    model = Model(one_dimensional=True, blob_factory=bf, num_blobs=1, verbose=False)
    assert model.geometry.Ny == 1
    assert model.geometry.Ly == 0
