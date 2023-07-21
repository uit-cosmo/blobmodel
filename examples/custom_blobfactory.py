from blobmodel import (
    Model,
    BlobFactory,
    Blob,
    show_model,
    AbstractBlobShape,
)
import numpy as np


# create custom class that inherits from BlobFactory
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
    Nx=100,
    Ny=100,
    Lx=2,
    Ly=2,
    dt=0.1,
    T=10,
    blob_shape="gauss",
    t_drain=2,
    periodic_y=True,
    num_blobs=1000,
    blob_factory=bf,
)

ds = tmp.make_realization(speed_up=True, error=1e-1)
show_model(dataset=ds)
