from blobmodel import (
    Model,
    BlobFactory,
    Blob,
    show_model,
    AbstractBlobShape,
)
import numpy as np
from typing import List


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
    ) -> List[Blob]:
        # set custom parameter distributions
        amp = [1, 1, 1]
        width = [2, 1, 1]
        vx = [1, 1, 1]
        vy = [0, 0, 0]

        posx = np.zeros(num_blobs)
        posy = [5, 2.5, 7.5]
        t_init = [0.0, 2.0, 5.0]

        # sort blobs by _t_init
        t_init = np.sort(t_init).tolist()

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
    Nx=64,
    Ny=64,
    Lx=10,
    Ly=10,
    dt=1,
    T=10,
    blob_shape="gauss",
    t_drain=100000,
    periodic_y=True,
    num_blobs=3,
    blob_factory=bf,
)

ds = tmp.make_realization(speed_up=True, error=1e-1)

import matplotlib.pyplot as plt

logo = ds.n.isel(t=7).values
plt.contourf(logo, 32)
plt.axis("off")
ax = plt.gca()
ax.set_aspect("equal", adjustable="box")
plt.savefig("logo.png", bbox_inches="tight")
plt.show()
