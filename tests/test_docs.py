# The following tests check that the docs examples run, if they fail make sure to change the docs too!


def test_getting_started(tmp_path, monkeypatch):
    # Write example.nc into a temp directory instead of the repo root. The
    # chdir happens outside the PLACEHOLDER markers so the docs snippet
    # (docs/getting_started.rst) is unaffected.
    monkeypatch.chdir(tmp_path)
    # PLACEHOLDER getting_started_0
    from blobmodel import Geometry, Model

    bm = Model(
        geometry=Geometry(
            Nx=10, Ny=10, Lx=10, Ly=10, dt=0.1, T=20, periodic_y=True, t_init=10
        ),
        num_blobs=100,
    )
    # PLACEHOLDER getting_started_1
    ds = bm.make_realization(file_name="example.nc")
    # PLACEHOLDER getting_started_2
    ds["n"].isel(y=0).mean(dim=("t")).plot()
    # PLACEHOLDER getting_started_3


def test_drainage_time():
    # PLACEHOLDER drainage_time_0
    import numpy as np
    from blobmodel import DefaultBlobFactory, Geometry, Model

    t_drain = np.linspace(2, 1, 100)

    tmp = Model(
        geometry=Geometry(Nx=100, Ny=1, Lx=10, Ly=0, dt=1, T=1000, periodic_y=False),
        blob_factory=DefaultBlobFactory(t_drain=t_drain),
        num_blobs=10,
    )
    tmp.make_realization()
    # PLACEHOLDER drainage_time_1


def test_one_dim():
    # PLACEHOLDER one_dim_0
    from blobmodel import (
        Geometry,
        Model,
        DefaultBlobFactory,
        DistributionEnum,
        BlobShapeImpl,
        BlobShapeEnum,
    )

    # Amplitudes are exponential and widths/velocities constant by default;
    # only the y-velocity needs to be set to zero for a 1D model.
    bf = DefaultBlobFactory(t_drain=10).set_sampler("vy", DistributionEnum.zeros)

    bm = Model(
        geometry=Geometry(Nx=100, Ny=1, Lx=10, Ly=0, dt=0.1, T=10, periodic_y=False),
        blob_shape=BlobShapeImpl(BlobShapeEnum.exp),
        num_blobs=20,
        blob_factory=bf,
        one_dimensional=True,
    )
    bm.make_realization(truncation_error=1e-2)
    # PLACEHOLDER one_dim_0


def test_blob_shapes():
    # PLACEHOLDER blob_shapes_0
    from blobmodel import (
        Geometry,
        Model,
        BlobShapeImpl,
        BlobShapeEnum,
        DefaultBlobFactory,
        DistributionEnum,
    )

    bm = Model(
        geometry=Geometry(Nx=100, Ny=100, Lx=10, Ly=10, dt=0.1, T=10, periodic_y=True),
        num_blobs=10,
        blob_shape=BlobShapeImpl(BlobShapeEnum.exp, BlobShapeEnum.lorentz),
    )
    # PLACEHOLDER blob_shapes_1
    bf = (
        DefaultBlobFactory()
        .set_sampler("amplitude", DistributionEnum.deg)
        .set_sampler("spp", DistributionEnum.deg, 0.5)
        .set_sampler("sps", DistributionEnum.deg, 0.5)
    )
    bm = Model(
        geometry=Geometry(Nx=100, Ny=100, Lx=10, Ly=10, dt=0.1, T=10),
        num_blobs=10,
        blob_shape=BlobShapeImpl(BlobShapeEnum.double_exp, BlobShapeEnum.double_exp),
        blob_factory=bf,
    )
    # PLACEHOLDER blob_shapes_2


def test_blob_tilt():
    # PLACEHOLDER blob_tilt_0
    from blobmodel import DefaultBlobFactory, DistributionEnum
    import numpy as np

    vx, vy = 1, 0
    wx, wy = 1, 1

    bf = (
        DefaultBlobFactory(blob_alignment=False)
        .set_sampler("amplitude", DistributionEnum.deg)
        .set_sampler("vx", DistributionEnum.deg, vx)
        .set_sampler("vy", DistributionEnum.deg, vy)
        .set_sampler("wp", DistributionEnum.deg, wx)
        .set_sampler("ws", DistributionEnum.deg, wy)
    )
    # PLACEHOLDER blob_tilt_1
    theta = np.pi / 2
    bf.set_theta_setter(lambda: theta)
    # PLACEHOLDER blob_tilt_2


def test_blob_labels():
    # PLACEHOLDER blob_labels_0
    from blobmodel import Geometry, Model, BlobShapeImpl, BlobShapeEnum

    bm = Model(
        geometry=Geometry(Nx=10, Ny=10, Lx=20, Ly=20, dt=0.1, T=20, periodic_y=True),
        blob_shape=BlobShapeImpl(BlobShapeEnum.gaussian),
        num_blobs=10,
        labels="individual",
        label_border=0.75,
    )

    ds = bm.make_realization(truncation_error=1e-2)

    ds["n"].isel(t=-1).plot()
    ds["blob_labels"].isel(t=-1).plot()
    # PLACEHOLDER blob_labels_1


def test_blob_factory():
    # PLACEHOLDER blob_factory_0
    from blobmodel import DefaultBlobFactory, DistributionEnum, Geometry, Model

    my_blob_factory = DefaultBlobFactory(t_drain=100).set_sampler(
        "amplitude", DistributionEnum.normal, free_parameter=5
    )

    bm = Model(
        geometry=Geometry(Nx=10, Ny=10, Lx=10, Ly=10, dt=0.1, T=20),
        blob_factory=my_blob_factory,
        num_blobs=100,
    )

    ds = bm.make_realization()
    # PLACEHOLDER blob_factory_1


def test_prebuilt_blobs():
    # PLACEHOLDER prebuilt_blobs_0
    from blobmodel import Blob, BlobShapeImpl, Geometry, Model
    import numpy as np

    blobs = [
        Blob(
            blob_id=i,
            blob_shape=BlobShapeImpl(),
            amplitude=1,
            width_p=1,
            width_s=1,
            v_x=1,
            v_y=0,
            pos_x0=0,
            pos_y0=pos_y0,
            t_init=t_init,
            t_drain=np.inf,  # no draining
        )
        for i, (pos_y0, t_init) in enumerate([(2, 0), (5, 2), (8, 4)])
    ]

    bm = Model.from_blobs(
        blobs, geometry=Geometry(Nx=10, Ny=10, Lx=10, Ly=10, dt=0.1, T=10)
    )
    ds = bm.make_realization()
    # PLACEHOLDER prebuilt_blobs_1


def test_callable_blob_factory():
    # PLACEHOLDER callable_blob_factory_0
    from blobmodel import Blob, BlobShapeImpl, CallableBlobFactory, Geometry, Model
    import numpy as np

    def blob_getter(rng: np.random.Generator) -> Blob:
        # Draw random numbers from the generator you are given (not from the
        # global np.random state) so that the realization is reproducible
        # through CallableBlobFactory(seed=...) or Model(seed=...).
        return Blob(
            blob_id=0,
            blob_shape=BlobShapeImpl(),
            amplitude=rng.exponential(),
            width_p=1,
            width_s=1,
            v_x=1,
            v_y=0,
            pos_x0=0,
            pos_y0=rng.uniform(0, 10),
            t_init=rng.uniform(0, 10),
            t_drain=np.inf,
        )

    bf = CallableBlobFactory(blob_getter, seed=42)

    bm = Model(
        geometry=Geometry(Nx=10, Ny=10, Lx=10, Ly=10, dt=0.1, T=10),
        num_blobs=10,  # blob_getter is called num_blobs times
        blob_factory=bf,
    )
    ds = bm.make_realization()
    # PLACEHOLDER callable_blob_factory_1


def test_custom_blob_factory():
    # PLACEHOLDER custom_blob_factory_0
    from blobmodel import BlobFactory, AbstractBlobShape, Blob, Geometry, Model
    import numpy as np

    class CustomBlobFactory(BlobFactory):
        def __init__(self, t_drain: float = np.inf, seed=None) -> None:
            # Blob draining is owned by the factory: each sampled Blob gets
            # this t_drain. Draw random numbers from self.rng: a seed passed
            # to Model replaces it (via BlobFactory.set_rng), making
            # realizations reproducible.
            self.t_drain = t_drain
            self.rng = np.random.default_rng(seed)

        def sample_blobs(
            self,
            Ly: float,
            T: float,
            num_blobs: int,
            blob_shape: AbstractBlobShape,
        ) -> list[Blob]:

            # set custom parameter distributions
            amp = np.linspace(0.01, 1, num=num_blobs)
            width = np.linspace(0.01, 1, num=num_blobs)
            vx = np.linspace(0.01, 1, num=num_blobs)
            vy = np.linspace(0.01, 1, num=num_blobs)

            posx = np.zeros(num_blobs)
            posy = self.rng.uniform(low=0.0, high=Ly, size=num_blobs)
            t_init = self.rng.uniform(low=0, high=T, size=num_blobs)

            # sort blobs by _t_init
            t_init = np.sort(t_init)

            return [
                Blob(
                    blob_id=i,
                    blob_shape=blob_shape,
                    amplitude=amp[i],
                    width_p=width[i],
                    width_s=width[i],
                    v_x=vx[i],
                    v_y=vy[i],
                    pos_x0=posx[i],
                    pos_y0=posy[i],
                    t_init=t_init[i],
                    t_drain=self.t_drain,
                )
                for i in range(num_blobs)
            ]

        def is_one_dimensional(self) -> bool:
            return False

    bf = CustomBlobFactory(t_drain=2)
    tmp = Model(
        geometry=Geometry(Nx=10, Ny=10, Lx=2, Ly=2, dt=0.1, T=10, periodic_y=True),
        num_blobs=10,
        blob_factory=bf,
    )

    ds = tmp.make_realization()
    # PLACEHOLDER custom_blob_factory_1


def test_burn_in():
    # PLACEHOLDER burn_in_0
    from blobmodel import Blob, CallableBlobFactory, Geometry, Model
    import numpy as np

    T = 10
    burn_in = 5  # roughly a few blob transit/drain times

    def blob_getter(rng: np.random.Generator) -> Blob:
        # Arrivals sampled on [-burn_in, T): blobs born before t = 0 have
        # already propagated into the domain when the output starts.
        return Blob(
            amplitude=rng.exponential(),
            pos_y0=rng.uniform(0, 10),
            t_init=rng.uniform(-burn_in, T),
        )

    bm = Model(
        geometry=Geometry(Nx=10, Ny=10, Lx=10, Ly=10, dt=0.1, T=T),
        # scale num_blobs with T + burn_in to keep the arrival rate fixed
        num_blobs=int(30 * (T + burn_in) / T),
        blob_factory=CallableBlobFactory(blob_getter, seed=42),
    )
    ds = bm.make_realization()
    # PLACEHOLDER burn_in_1


def test_custom_sampler():
    # PLACEHOLDER custom_sampler_0
    from blobmodel import DefaultBlobFactory, Geometry, Model

    # Draw the widths from a log-normal distribution, which is not built in.
    # The callable receives the factory's random number generator (draw from
    # it, not from the global np.random state, to stay seedable) and the
    # number of blobs, and returns one value per blob.
    bf = DefaultBlobFactory(seed=42).set_sampler(
        "wp", lambda rng, num_blobs: rng.lognormal(sigma=0.5, size=num_blobs)
    )

    bm = Model(
        geometry=Geometry(Nx=10, Ny=10, Lx=10, Ly=10, dt=0.1, T=20),
        blob_factory=bf,
        num_blobs=100,
    )
    ds = bm.make_realization()
    # PLACEHOLDER custom_sampler_1
