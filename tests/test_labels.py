import pytest
from blobmodel import Geometry, Model, BlobFactory, Blob, AbstractBlobShape
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
    ) -> List[Blob]:
        return [
            Blob(
                blob_id=i,
                blob_shape=blob_shape,
                amplitude=1,
                width_p=1,
                width_s=1,
                v_x=1,
                v_y=0,
                pos_x0=0,
                pos_y0=Ly / 2,
                t_init=0,
                t_drain=np.inf,
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
        geometry=Geometry(Nx=5, Ny=1, Lx=5, Ly=5, dt=1, T=5, periodic_y=True),
        num_blobs=1,
        blob_factory=bf,
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


def test_individual_labels_ignore_blob_id():
    """
    labels="individual" assigns labels 1..num_blobs from each blob's position
    in the factory output, so hand-built blobs sharing the default blob_id=0
    still get distinct labels.
    """
    warnings.filterwarnings("ignore")
    blobs = [Blob(t_init=0), Blob(t_init=2)]
    assert blobs[0].blob_id == blobs[1].blob_id == 0
    bm = Model.from_blobs(
        blobs=blobs,
        geometry=Geometry(Nx=5, Ny=1, Lx=5, Ly=5, dt=1, T=5, periodic_y=True),
        labels="individual",
        verbose=False,
    )
    ds = bm.make_realization(speed_up=False)
    labels_found = set(np.unique(ds["blob_labels"].values))
    assert {1.0, 2.0} <= labels_found
