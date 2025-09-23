import blobmodel as bm
import superposedpulses as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.signal import welch

import cosmoplots as cp

matplotlib_params = plt.rcParams
cp.set_rcparams_dynamo(matplotlib_params, 1)
plt.rcParams.update(matplotlib_params)


dt = 0.1
T = int(1e5)
num_blobs = int(1e5)
start_index = int(100 / dt)

# Blobmodel
model = bm.Model(
    Nx=1,
    Ny=1,
    Lx=1,
    Ly=0,
    dt=dt,
    T=T,
    num_blobs=num_blobs,
    blob_shape=bm.BlobShapeImpl(bm.BlobShapeEnum.exp, bm.BlobShapeEnum.gaussian),
    periodic_y=False,
    t_drain=1e10,
    blob_factory=bm.DefaultBlobFactory(
        A_dist=bm.DistributionEnum.exp, vy_dist=bm.DistributionEnum.zeros
    ),
    verbose=True,
    t_init=0,
    one_dimensional=True,  # Sets Ly = 0, Ny = 1, and checks vy = 0, and sets the y blob shape to 1.
)
ds = model.make_realization(speed_up=True, error=1e-10)
ds = ds.isel(t=slice(start_index, int(1e50)))
bm_signal = ds.n.isel(x=0, y=0).values

# Superposedpulses
waiting_time = T / num_blobs
model = sp.PointModel(waiting_time=waiting_time, total_duration=T, dt=dt)
model.set_amplitude_distribution("exp")
times, signal = model.make_realization()
times, signal = times[start_index:], signal[start_index:]

average_amplitude = 1
average_squared_amplitude = 2
integral_exponential_pulse_shape = 1
integral_square_exponential_pulse_shape = 0.5
model_gamma = 1 / waiting_time
mean_theory = model_gamma * average_amplitude * integral_exponential_pulse_shape
rms_squared_theory = (
    model_gamma * average_squared_amplitude * integral_square_exponential_pulse_shape
)


print("BM mean: {:.4f}".format(ds.n.mean(dim="t").item()))
print("SP mean: {:.4f}".format(np.mean(signal)))
print("A mean: {:.4f}".format(mean_theory))

print("BM rms: {:.4f}".format(np.mean((bm_signal - np.mean(bm_signal)) ** 2)))
print("SP rms: {:.4f}".format(np.mean((signal - np.mean(signal)) ** 2)))
print("A rms: {:.4f}".format(rms_squared_theory))

fig, ax = plt.subplots(figsize=(4, 3))

ax.plot(ds.t.values, bm_signal, color="blue", label="BM")
ax.plot(times, signal, color="green", label="SP")
ax.legend()

plt.show()

# Compute histograms with numpy
bins = 100
bm_hist, bm_bins = np.histogram(bm_signal, bins=bins, density=True)
sp_hist, sp_bins = np.histogram(signal, bins=bins, density=True)

# Use bin centers for plotting
bm_bin_centers = (bm_bins[:-1] + bm_bins[1:]) / 2
sp_bin_centers = (sp_bins[:-1] + sp_bins[1:]) / 2
analytical = (
    1
    / gamma(model_gamma)
    * sp_bin_centers ** (model_gamma - 1)
    * np.exp(-sp_bin_centers)
)

# Plot histograms with ax.plot
fig, ax = plt.subplots(figsize=(4, 3))
ax.scatter(bm_bin_centers, bm_hist, color="blue", label="BM")
ax.scatter(sp_bin_centers, sp_hist, color="green", label="SP")
ax.plot(sp_bin_centers, analytical, color="black", ls="--", label="A")

ax.set_xlabel(r"$\Phi$")
ax.set_ylabel(r"$P_\Phi(\Phi)$")
ax.set_title("Histogram of BM and SP Signals")
ax.set_yscale("log")
ax.set_ylim((1e-6, 1e2))
ax.legend()
plt.show()

# PSD
fig, ax = plt.subplots(figsize=(4, 3))

bm_freqs, bm_psd = welch(
    (bm_signal - bm_signal.mean()) / bm_signal.std(), fs=1 / dt, nperseg=1000
)
bm_freqs, bm_psd = bm_freqs[1:], bm_psd[1:]  # Remove zero-frequency
bm_freqs = 2 * np.pi * bm_freqs

sp_freqs, sp_psd = welch(
    (signal - signal.mean()) / signal.std(), fs=1 / dt, nperseg=1000
)
sp_freqs, sp_psd = sp_freqs[1:], sp_psd[1:]  # Remove zero-frequency
sp_freqs = 2 * np.pi * sp_freqs

analytical_psd = 4 / (1 + sp_freqs**2)

ax.plot(bm_freqs, bm_psd, color="blue", label="BM")
ax.plot(sp_freqs, sp_psd, color="green", label="SP")
ax.plot(sp_freqs, analytical_psd, color="black", ls="--", label="A")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\omega$")
ax.set_ylabel(r"$\Omega_{\widetilde{\Phi}}(\omega)$")
ax.set_title("PSD")

plt.show()
