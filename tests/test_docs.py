# The following tests check that the docs examples run, if they fail make sure to change the docs too!


def test_getting_started():
    # PLACEHOLDER getting_started_0
    from blobmodel import Model

    bm = Model(
        Nx=10,
        Ny=10,
        Lx=10,
        Ly=10,
        dt=0.1,
        T=20,
        periodic_y=True,
        num_blobs=100,
        t_drain=1e10,
        t_init=10,
    )
    # PLACEHOLDER getting_started_1
    ds = bm.make_realization(file_name="example.nc")
    # PLACEHOLDER getting_started_2
    ds["n"].isel(y=0).mean(dim=("t")).plot()
    # PLACEHOLDER getting_started_3


def test_drainage_time():
    # PLACEHOLDER drainage_time_0
    import numpy as np
    from blobmodel import Model

    t_drain = np.linspace(2, 1, 100)

    tmp = Model(
        Nx=100,
        Ny=1,
        Lx=10,
        Ly=0,
        dt=1,
        T=1000,
        t_drain=t_drain,
        periodic_y=False,
        num_blobs=10,
    )
    tmp.make_realization()
    # PLACEHOLDER drainage_time_1


def test_one_dim():
    # PLACEHOLDER one_dim_0
    from blobmodel import (
        Model,
        DefaultBlobFactory,
        DistributionEnum,
        BlobShapeImpl,
        BlobShapeEnum,
    )

    bf = DefaultBlobFactory(
        A_dist=DistributionEnum.exp,
        wx_dist=DistributionEnum.deg,
        vx_dist=DistributionEnum.deg,
        vy_dist=DistributionEnum.zeros,
    )

    bm = Model(
        Nx=100,
        Ny=1,
        Lx=10,
        Ly=0,
        dt=0.1,
        T=10,
        periodic_y=False,
        blob_shape=BlobShapeImpl(BlobShapeEnum.exp),
        num_blobs=20,
        t_drain=10,
        blob_factory=bf,
        one_dimensional=True,
    )
    bm.make_realization(speed_up=True, error=1e-2)
    # PLACEHOLDER one_dim_0


def test_blob_shapes():
    # PLACEHOLDER blob_shapes_0
    from blobmodel import (
        Model,
        BlobShapeImpl,
        BlobShapeEnum,
        DefaultBlobFactory,
        DistributionEnum,
    )

    bm = Model(
        Nx=100,
        Ny=100,
        Lx=10,
        Ly=10,
        dt=0.1,
        T=10,
        num_blobs=10,
        blob_shape=BlobShapeImpl(BlobShapeEnum.exp, BlobShapeEnum.lorentz),
        periodic_y=True,
        t_drain=1e10,
    )
    # PLACEHOLDER blob_shapes_1
    bf = DefaultBlobFactory(
        A_dist=DistributionEnum.deg,
        wx_dist=DistributionEnum.deg,
        spx_dist=DistributionEnum.deg,
        spy_dist=DistributionEnum.deg,
        shape_param_x_parameter=0.5,
        shape_param_y_parameter=0.5,
    )
    bm = Model(
        Nx=100,
        Ny=100,
        Lx=10,
        Ly=10,
        dt=0.1,
        T=10,
        num_blobs=10,
        blob_shape=BlobShapeImpl(BlobShapeEnum.double_exp, BlobShapeEnum.double_exp),
        t_drain=1e10,
        blob_factory=bf,
    )
    # PLACEHOLDER blob_shapes_2


def test_blob_tilt():
    # PLACEHOLDER blob_tilt_0
    from blobmodel import DefaultBlobFactory, DistributionEnum
    import numpy as np

    vx, vy = 1, 0
    wx, wy = 1, 1

    bf = DefaultBlobFactory(
        A_dist=DistributionEnum.deg,
        vy_parameter=vy,
        vx_parameter=vx,
        wx_parameter=wx,
        wy_parameter=wy,
        blob_alignment=False,
    )
    # PLACEHOLDER blob_tilt_1
    theta = np.pi / 2
    bf.set_theta_setter(lambda: theta)
    # PLACEHOLDER blob_tilt_2


def test_blob_labels():
    # PLACEHOLDER blob_labels_0
    from blobmodel import Model, BlobShapeImpl, BlobShapeEnum

    bm = Model(
        Nx=10,
        Ny=10,
        Lx=20,
        Ly=20,
        dt=0.1,
        T=20,
        periodic_y=True,
        blob_shape=BlobShapeImpl(BlobShapeEnum.gaussian),
        num_blobs=10,
        t_drain=1e10,
        labels="individual",
        label_border=0.75,
    )

    ds = bm.make_realization(speed_up=True, error=1e-2)

    ds["n"].isel(t=-1).plot()
    ds["blob_labels"].isel(t=-1).plot()
    # PLACEHOLDER blob_labels_1


def test_blob_factory():
    # PLACEHOLDER blob_factory_0
    from blobmodel import DefaultBlobFactory, DistributionEnum, Model

    my_blob_factory = DefaultBlobFactory(A_dist=DistributionEnum.normal, A_parameter=5)

    bm = Model(
        Nx=10,
        Ny=10,
        Lx=10,
        Ly=10,
        dt=0.1,
        T=20,
        blob_factory=my_blob_factory,
        t_drain=100,
        num_blobs=100,
    )

    ds = bm.make_realization()
    # PLACEHOLDER blob_factory_1


def test_custom_blob_factory():
    # PLACEHOLDER custom_blob_factory_0
    from blobmodel import BlobFactory, AbstractBlobShape, Blob, Model
    import numpy as np

    class CustomBlobFactory(BlobFactory):
        def __init__(self) -> None:
            pass

        def sample_blobs(
            self,
            Ly: float,
            T: float,
            num_blobs: int,
            blob_shape: AbstractBlobShape,
            t_drain: float,
        ) -> list[Blob]:

            # set custom parameter distributions
            amp = np.linspace(0.01, 1, num=num_blobs)
            width = np.linspace(0.01, 1, num=num_blobs)
            vx = np.linspace(0.01, 1, num=num_blobs)
            vy = np.linspace(0.01, 1, num=num_blobs)

            posx = np.zeros(num_blobs)
            posy = np.random.uniform(low=0.0, high=Ly, size=num_blobs)
            t_init = np.random.uniform(low=0, high=T, size=num_blobs)

            # sort blobs by _t_init
            t_init = np.sort(t_init)

            return [
                Blob(
                    blob_id=i,
                    blob_shape=blob_shape,
                    amplitude=amp[i],
                    width_prop=width[i],
                    width_perp=width[i],
                    v_x=vx[i],
                    v_y=vy[i],
                    pos_x=posx[i],
                    pos_y=posy[i],
                    t_init=t_init[i],
                    t_drain=t_drain,
                )
                for i in range(num_blobs)
            ]

        def is_one_dimensional(self) -> bool:
            return False

    bf = CustomBlobFactory()
    tmp = Model(
        Nx=10,
        Ny=10,
        Lx=2,
        Ly=2,
        dt=0.1,
        T=10,
        t_drain=2,
        periodic_y=True,
        num_blobs=10,
        blob_factory=bf,
    )

    ds = tmp.make_realization()
    # PLACEHOLDER custom_blob_factory_1
