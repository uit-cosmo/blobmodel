import pytest
from blobmodel import Model, BlobFactory, Blob, BlobShapeImpl, AbstractBlobShape
import numpy as np
import warnings
from typing import List


# here you can define your custom parameter distributions
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
    ) -> List[Blob]:
        # set custom parameter distributions
        _amp = np.ones(num_blobs)
        _width = np.ones(num_blobs)
        _vx = np.ones(num_blobs)
        _vy = np.zeros(num_blobs)

        _posx = np.zeros(num_blobs)
        _posy = np.ones(num_blobs) * Ly / 2
        _t_init = np.ones(num_blobs) * 0

        return [
            Blob(
                blob_id=i,
                blob_shape=blob_shape,
                amplitude=_amp[i],
                width_prop=_width[i],
                width_perp=_width[i],
                v_x=_vx[i],
                v_y=_vy[i],
                pos_x=_posx[i],
                pos_y=_posy[i],
                t_init=_t_init[i],
                t_drain=t_drain,
            )
            for i in range(num_blobs)
        ]

    def is_one_dimensional(self) -> bool:
        return False


def test_bloblabels_speedup():
    warnings.filterwarnings("ignore")
    bf = CustomBlobFactory()
    bm = Model(
        Nx=5,
        Ny=1,
        Lx=5,
        Ly=5,
        dt=1,
        T=5,
        periodic_y=True,
        blob_shape=BlobShapeImpl("gauss"),
        num_blobs=1,
        blob_factory=bf,
        t_drain=1e10,
        labels="same",
    )
    ds = bm.make_realization(speed_up=True, error=1e-2)
    correct_labels = np.array(
        [
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ]
    )
    diff = ds["blob_labels"].values - correct_labels
    assert np.max(diff) < 0.00001


@pytest.mark.parametrize("labels", [("individual"), ("same")])
def test_bloblabels(labels):
    warnings.filterwarnings("ignore")
    bf = CustomBlobFactory()
    bm = Model(
        Nx=5,
        Ny=1,
        Lx=5,
        Ly=5,
        dt=1,
        T=5,
        periodic_y=True,
        blob_shape=BlobShapeImpl("gauss"),
        num_blobs=1,
        blob_factory=bf,
        t_drain=1e10,
        labels=labels,
    )
    ds = bm.make_realization(speed_up=False)
    correct_labels = np.array(
        [
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ]
    )
    diff = ds["blob_labels"].values - correct_labels
    assert np.max(diff) < 0.00001
