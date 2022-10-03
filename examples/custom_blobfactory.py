from blobmodel import Model, BlobFactory, Blob, show_model
import numpy as np

# create custom class that inherits from BlobFactory
# here you can define your custom parameter distributions
class CustomBlobFactory(BlobFactory):
    def __init__(self) -> None:
        pass

    def sample_blobs(
        self, Ly: float, T: float, num_blobs: int, blob_shape: str, t_drain: float
    ) -> list[Blob]:

        # set custom parameter distributions
        _amp = np.linspace(0.01, 1, num=num_blobs)
        _width = np.linspace(0.01, 1, num=num_blobs)
        _vx = np.linspace(0.01, 1, num=num_blobs)
        _vy = np.linspace(0.01, 1, num=num_blobs)

        _posx = np.zeros(num_blobs)
        _posy = np.random.uniform(low=0.0, high=Ly, size=num_blobs)
        _t_init = np.random.uniform(low=0, high=T, size=num_blobs)

        # sort blobs by _t_init
        _t_init = np.sort(_t_init)

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
