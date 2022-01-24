from blobmodel import Model, BlobFactory, Blob
import numpy as np
import warnings


# here you can define your custom parameter distributions
class CustomBlobFactory(BlobFactory):
    def __init__(self) -> None:
        pass

    def sample_blobs(
        self, Ly: float, T: float, num_blobs: int, blob_shape: str, t_drain: float
    ) -> list[Blob]:

        # set custom parameter distributions
        __amp = np.ones(num_blobs)
        __width = np.ones(num_blobs)
        __vx = np.ones(num_blobs)
        __vy = np.zeros(num_blobs)

        __posx = np.zeros(num_blobs)
        __posy = np.ones(num_blobs) * Ly / 2
        __t_init = np.ones(num_blobs) * 0

        return [
            Blob(
                id=i,
                blob_shape=blob_shape,
                amplitude=__amp[i],
                width_prop=__width[i],
                width_perp=__width[i],
                v_x=__vx[i],
                v_y=__vy[i],
                pos_x=__posx[i],
                pos_y=__posy[i],
                t_init=__t_init[i],
                t_drain=t_drain,
            )
            for i in range(num_blobs)
        ]


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
        blob_shape="gauss",
        num_blobs=1,
        blob_factory=bf,
        t_drain=1e10,
    )
    ds = bm.make_realization(speed_up=True, error=1e-2, labels=True)
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


def test_bloblabels():
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
        blob_shape="gauss",
        num_blobs=1,
        blob_factory=bf,
        t_drain=1e10,
    )
    ds = bm.make_realization(speed_up=False, labels=True)
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
