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
t_drain = np.linspace(2, 1, 10)

bf = (
    DefaultBlobFactory(t_drain=t_drain)
    .set_sampler("amplitude", DistributionEnum.deg)
    .set_sampler("vy", DistributionEnum.zeros)
)

tmp = Model(
    geometry=Geometry(Nx=10, Ny=1, Lx=10, Ly=0, dt=1, T=1000, periodic_y=False),
    blob_shape=BlobShapeImpl(BlobShapeEnum.exp, BlobShapeEnum.gaussian),
    num_blobs=10000,
    blob_factory=bf,
)


def test_decreasing_t_drain():
    """
    Checks that models with variable t_drain run and lead to a results lower than the constant case.
    """
    ds = tmp.make_realization(truncation_error=1e-2)
    model_profile = ds.n.mean(dim=("t"))

    x = np.linspace(0, 10, 10)
    t_p = 1
    t_w = 1 / 10
    amp = 1
    v_p = 1.0
    t_loss = 2.0
    t_d = t_loss * t_p / (t_loss + t_p)

    analytical_profile = (
        1 / np.sqrt(np.pi) * t_d / t_w * amp * np.exp(-x / (v_p * t_loss))
    )
    assert (model_profile.values[2:] < analytical_profile[2:]).all()
