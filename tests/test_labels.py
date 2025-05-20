import pytest
from blobmodel import Model, BlobFactory, Blob, AbstractBlobShape
import numpy as np
import warnings
from typing import List


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
        return [
            Blob(
                blob_id=i,
                blob_shape=blob_shape,
                amplitude=1,
                width_prop=1,
                width_perp=1,
                v_x=1,
                v_y=0,
                pos_x=0,
                pos_y=Ly / 2,
                t_init=0,
                t_drain=t_drain,
            )
            for i in range(num_blobs)
        ]

    def is_one_dimensional(self) -> bool:
        return False


@pytest.mark.parametrize("labels", ["individual", "same"])
@pytest.mark.parametrize("speed_up", [True, False])
def test_bloblabels(labels, speed_up):
    """
    Checks correct blob labels with a single blob.
    """
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
        num_blobs=1,
        blob_factory=bf,
        t_drain=1e10,
        labels=labels,
    )
    ds = bm.make_realization(speed_up=speed_up)
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
