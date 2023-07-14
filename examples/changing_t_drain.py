from blobmodel import Model, DefaultBlobFactory
import matplotlib.pyplot as plt
import numpy as np

bf = DefaultBlobFactory(A_dist="deg", wx_dist="deg", vx_dist="deg", vy_dist="zeros")

t_drain = np.linspace(2, 1, 100)

tmp = Model(
    Nx=100,
    Ny=1,
    Lx=10,
    Ly=0,
    dt=1,
    T=1000,
    blob_shape="exp",
    t_drain=t_drain,
    periodic_y=False,
    num_blobs=10000,
    blob_factory=bf,
)

ds = tmp.make_realization(file_name="profile_comparison.nc", speed_up=True, error=1e-2)


def plot_cahnging_t_drain(ds):
    x = np.linspace(0, 10, 100)
    t_p = 1
    t_w = 1 / 10
    amp = 1
    v_p = 1.0
    t_loss = t_drain
    t_d = t_loss * t_p / (t_loss + t_p)

    analytical_profile = (
        1 / np.sqrt(np.pi) * t_d / t_w * amp * np.exp(-x / (v_p * t_loss))
    )

    ds.n.isel(y=0).mean(dim=("t")).plot(label="decreasing t_drain")
    plt.yscale("log")
    plt.plot(x, analytical_profile, label="constant t_drain")
    plt.legend()
    plt.show()


plot_cahnging_t_drain(ds)
