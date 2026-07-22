from blobmodel import Geometry, Model, BlobShapeImpl


def test_geometry_str():
    """
    Tests geometry string function.
    """
    geo = Geometry(1, 1, 1, 1, 1, 1, 1, False)
    assert (
        str(geo)
        == "Geometry parameters:  Nx:1,  Ny:1, Lx:1, Ly:1, dt:1, T:1, t_init:1, y-periodicity:False, "
        + "x0:0, y0:0"
    )


def test_model_str():
    """
    Tests model string function.
    """
    bm = Model(
        geometry=Geometry(Nx=2, Ny=2, Lx=10, Ly=10, dt=0.5, T=1, periodic_y=False),
        blob_shape=BlobShapeImpl(),
        num_blobs=1,
    )
    assert str(bm) == "2d Blob Model with num_blobs:1 and t_drain:10"
