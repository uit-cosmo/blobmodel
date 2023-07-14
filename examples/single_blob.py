from blobmodel import Model, show_model, BlobFactory, Blob, AbstractBlobShape
import numpy as np


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
        amp = np.ones(num_blobs)
        width = np.ones(num_blobs)
        vx = np.ones(num_blobs)
        vy = np.ones(num_blobs)

        posx = np.zeros(num_blobs)
        posy = np.ones(num_blobs) * Ly / 2
        t_init = np.ones(num_blobs) * 0

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

bm = Model(
    Nx=100,
    Ny=100,
    Lx=10,
    Ly=10,
    dt=0.1,
    T=10,
    periodic_y=True,
    blob_shape="exp",
    num_blobs=1,
    blob_factory=bf,
    t_drain=1e10,
)

# create data
ds = bm.make_realization(speed_up=True, error=1e-2)

# show animation and save as gif
show_model(dataset=ds, interval=100, gif_name="example.gif", fps=10)
