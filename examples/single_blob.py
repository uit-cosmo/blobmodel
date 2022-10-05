from blobmodel import Model, show_model, BlobFactory, Blob
import numpy as np

# here you can define your custom parameter distributions
class CustomBlobFactory(BlobFactory):
    def __init__(self) -> None:
        pass

    def sample_blobs(
        self, Ly: float, T: float, num_blobs: int, blob_shape: str, t_drain: float
    ) -> list[Blob]:

        # set custom parameter distributions
        _amp = np.ones(num_blobs)
        _width = np.ones(num_blobs)
        _vx = np.ones(num_blobs)
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
show_model(dataset=ds, interval=100, save=True, gif_name="example.gif", fps=10)
