from blobmodel import Blob, BlobShapeEnum, BlobShapeImpl, Geometry, Model
import matplotlib.pyplot as plt

geometry = Geometry(Nx=64, Ny=1, Lx=8, Ly=0, dt=0.05, T=16)
blob_shape = BlobShapeImpl(BlobShapeEnum.gaussian)


def time_trace(**kwargs):
    """Signal at x = 4 of a single blob that passes there at t = 8."""
    blob = Blob(
        blob_shape=blob_shape,
        amplitude=1,
        width_p=2,
        v_x=1,
        pos_x0=4,
        t_init=8,
        **kwargs,
    )
    model = Model.from_blobs(
        [blob], geometry=geometry, one_dimensional=True, verbose=False
    )
    return model.make_realization(speed_up=False).n.isel(x=32)


cases = [
    ({}, "no draining, no lifetime"),
    ({"t_drain": 3}, "t_drain = 3"),
    ({"t_lifetime": 2}, "t_lifetime = 2"),
    ({"t_drain": 3, "t_lifetime": 2}, "both"),
]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for kwargs, label in cases:
    trace = time_trace(**kwargs)
    for ax in axes:
        trace.plot(ax=ax, label=label)
axes[1].set_yscale("log")
axes[1].set_ylim(1e-4, 1)
for ax in axes:
    ax.set_title("")
    ax.set_ylabel("n")
axes[0].legend()
plt.tight_layout()
plt.savefig("pulse_lifetime_plot.png", dpi=150, bbox_inches="tight")
plt.show()
