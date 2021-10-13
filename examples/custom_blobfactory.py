from blobmodel import Model, BlobFactory, Blob, show_model, Geometry
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
        __amp = np.linspace(0.01, 1, num=num_blobs)
        __width = np.linspace(0.01, 1, num=num_blobs)
        __vx = np.linspace(0.01, 1, num=num_blobs)
        __vy = np.linspace(0.01, 1, num=num_blobs)

        __posx = np.zeros(num_blobs)
        __posy = np.random.uniform(low=0.0, high=Ly, size=num_blobs)
        __t_init = np.random.uniform(low=0, high=T, size=num_blobs)

        # sort blobs by __t_init
        __t_init = np.sort(__t_init)

        # this block must remain the same
        __blobs = []
        for i in range(num_blobs):
            __blobs.append(
                Blob(
                    id=i,
                    blob_shape=blob_shape,
                    amplitude=__amp[i],
                    width_x=__width[i],
                    width_y=__width[i],
                    v_x=__vx[i],
                    v_y=__vy[i],
                    pos_x=__posx[i],
                    pos_y=__posy[i],
                    t_init=__t_init[i],
                    t_drain=t_drain,
                )
            )
        return __blobs


geo = Geometry(Nx=100, Ny=100, Lx=2, Ly=2, dt=0.1, T=10, periodic_y=True,)
bf = CustomBlobFactory()
tmp = Model(
    geometry=geo, blob_shape="gauss", t_drain=2, num_blobs=1000, blob_factory=bf,
)

ds = tmp.integrate(speed_up=True, truncation_Lx=1)
show_model(ds=ds)
