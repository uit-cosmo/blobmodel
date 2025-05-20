from blobmodel import (
    Model,
    DefaultBlobFactory,
    DistributionEnum,
    BlobShapeImpl,
    BlobShapeEnum,
)
import numpy as np


# use DefaultBlobFactory to define distribution functions fo random variables
bf = DefaultBlobFactory(A_dist=DistributionEnum.deg, vy_dist=DistributionEnum.zeros)

t_drain = np.linspace(2, 1, 10)

tmp = Model(
    Nx=10,
    Ny=1,
    Lx=10,
    Ly=0,
    dt=1,
    T=1000,
    blob_shape=BlobShapeImpl(BlobShapeEnum.exp, BlobShapeEnum.gaussian),
    t_drain=t_drain,
    periodic_y=False,
    num_blobs=10000,
    blob_factory=bf,
)


def test_decreasing_t_drain():
    """
    Checks that models with variable t_drain run and lead to a results lower than the constant case.
    """
    ds = tmp.make_realization(speed_up=True, error=1e-2)
    model_profile = ds.n.isel(y=0).mean(dim=("t"))

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
