import numpy as np
import pytest
import xarray as xr
from blobmodel import Geometry, Model, to_imaging_dataset


def make_model(labels="off", one_dimensional=False):
    geometry = (
        Geometry(Nx=4, Ny=1, Lx=4, Ly=0, dt=1, T=5, periodic_y=False)
        if one_dimensional
        else Geometry(Nx=4, Ny=3, Lx=4, Ly=3, dt=1, T=5, periodic_y=False)
    )
    return Model(
        geometry=geometry,
        num_blobs=2,
        labels=labels,
        one_dimensional=one_dimensional,
        verbose=False,
        seed=42,
    )


def test_imaging_layout_matches_downstream_conversion():
    """
    layout="imaging" returns exactly the dataset downstream repos build by
    hand from the default layout: frames(y, x, time) with meshgrid R/Z
    coordinates.
    """
    ds = make_model().make_realization(truncation_error=1e-10)
    grid_r, grid_z = np.meshgrid(ds.x.values, ds.y.values)
    expected = xr.Dataset(
        {"frames": (["y", "x", "time"], ds.n.values)},
        coords={
            "R": (["y", "x"], grid_r),
            "Z": (["y", "x"], grid_z),
            "time": (["time"], ds.t.values),
        },
    )

    ds_imaging = make_model().make_realization(truncation_error=1e-10, layout="imaging")
    xr.testing.assert_allclose(ds_imaging, expected)


def test_to_imaging_dataset_converter():
    """
    The standalone converter produces the same dataset as the layout
    argument, so already-computed datasets can be converted too.
    """
    ds = make_model().make_realization(truncation_error=1e-10)
    ds_imaging = make_model().make_realization(truncation_error=1e-10, layout="imaging")
    xr.testing.assert_allclose(to_imaging_dataset(ds), ds_imaging)


def test_imaging_layout_keeps_blob_labels():
    """
    A blob_labels variable is carried over to the imaging layout with the
    renamed time dimension.
    """
    ds_imaging = make_model(labels="individual").make_realization(layout="imaging")
    assert "blob_labels" in ds_imaging
    assert ds_imaging.blob_labels.dims == ("y", "x", "time")


def test_invalid_layout_raises():
    with pytest.raises(ValueError, match="layout"):
        make_model().make_realization(layout="bogus")


def test_imaging_layout_rejects_one_dimensional():
    """
    The imaging format is inherently two-dimensional; both the model argument
    and the converter reject 1D output.
    """
    model = make_model(one_dimensional=True)
    with pytest.raises(ValueError, match="two-dimensional"):
        model.make_realization(layout="imaging")

    ds_1d = model.make_realization()
    with pytest.raises(ValueError, match="two-dimensional"):
        to_imaging_dataset(ds_1d)
