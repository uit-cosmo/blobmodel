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
        __amp = np.ones(num_blobs)
        __width = np.ones(num_blobs)
        __vx = np.ones(num_blobs)
        __vy = np.ones(num_blobs) * 10

        __posx = np.zeros(num_blobs)
        __posy = np.ones(num_blobs) * Ly / 2
        __t_init = np.ones(num_blobs) * 0

        # this block must remain the same
        __blobs = []
        for i in range(num_blobs):
            __blobs.append(
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
            )
        return __blobs


# bf = CustomBlobFactory()

bm = Model(
    Nx=100,
    Ny=100,
    Lx=10,
    Ly=10,
    dt=0.1,
    T=10,
    periodic_y=True,
    blob_shape="exp",
    num_blobs=100,
    # blob_factory=bf,
)

# create data
ds = bm.make_realization(speed_up=True, truncation_Lx=2)

# show animation and save as gif
show_model(ds=ds, interval=100, save=True, gif_name="example.gif", fps=10)
