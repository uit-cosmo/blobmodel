from blobmodel import Model, BlobFactory, Blob, AbstractBlobShape, BlobShapeImpl
import numpy as np
from typing import List


class CustomBlobFactoryVy0(BlobFactory):
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


class CustomBlobFactoryVx0(BlobFactory):
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
        _vx = np.zeros(num_blobs)
        _vy = np.ones(num_blobs)

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


bf_vy_0 = CustomBlobFactoryVy0()

bm_vy_0 = Model(
    Nx=10,
    Ny=10,
    Lx=10,
    Ly=10,
    dt=1,
    T=1,
    periodic_y=True,
    num_blobs=1,
    blob_factory=bf_vy_0,
    t_drain=1e10,
)

bf_vx_0 = CustomBlobFactoryVx0()

bm_vx_0 = Model(
    Nx=10,
    Ny=10,
    Lx=10,
    Ly=10,
    dt=1,
    T=1,
    periodic_y=True,
    num_blobs=1,
    blob_factory=bf_vx_0,
    t_drain=1e10,
)


def test_vy_0():
    assert bm_vy_0.make_realization(speed_up=True, error=1e-2)


def test_vx_0():
    assert bm_vx_0.make_realization(speed_up=True, error=1e-2)
