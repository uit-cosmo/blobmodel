import blobmodel as bm
import superposedpulses as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.signal import welch

dt = 0.1
T = 100000
num_blobs = 100000
start_index = int(100 / dt)

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
    ),  # DeterministicBlobFactory(blobs),
    verbose=True,
    t_init=0,
    one_dimensional=True,
)
ds = model.make_realization(speed_up=True, error=1e-10)

waiting_time = T / num_blobs
model_gamma = 1 / waiting_time
average_amplitude = 1
average_squared_amplitude = 2
integral_exponential_pulse_shape = 1
integral_square_exponential_pulse_shape = 0.5

model = sp.PointModel(waiting_time=waiting_time, total_duration=T, dt=dt)
# model.set_custom_forcing_generator(ForcingGenerator())
model.set_amplitude_distribution("exp")
times, signal = model.make_realization()
times, signal = times[start_index:], signal[start_index:]
ds = ds.isel(t=slice(start_index, int(1e50)))

mean_theory = model_gamma * average_amplitude * integral_exponential_pulse_shape
rms_squared_theory = (
    model_gamma * average_squared_amplitude * integral_square_exponential_pulse_shape
)

mean_realization = np.mean(signal)
bm_signal = ds.n.isel(x=0, y=0).values

print("BM mean: {:.4f}".format(ds.n.mean(dim="t").item()))
print("SP mean: {:.4f}".format(np.mean(signal)))
print("A mean: {:.4f}".format(mean_theory))

print("BM rms: {:.4f}".format(np.mean((bm_signal - np.mean(bm_signal)) ** 2)))
print("SP rms: {:.4f}".format(np.mean((signal - np.mean(signal)) ** 2)))
print("A rms: {:.4f}".format(rms_squared_theory))

fig, ax = plt.subplots()

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
fig, ax = plt.subplots()
ax.scatter(bm_bin_centers, bm_hist, color="blue", label="BM")
ax.scatter(sp_bin_centers, sp_hist, color="green", label="SP")
ax.plot(sp_bin_centers, analytical, color="black", ls="--", label="A")

print(np.sum(bm_hist * bm_bin_centers**2) * np.diff(bm_bin_centers)[0])
print(np.sum(sp_hist * sp_bin_centers**2) * np.diff(sp_bin_centers)[0])
print(np.sum(analytical * sp_bin_centers**2) * np.diff(sp_bin_centers)[0])
ax.set_xlabel("Signal Value")
ax.set_ylabel("Density")
ax.set_title("Histogram of BM and SP Signals")
ax.set_yscale("log")
ax.set_ylim((1e-6, 1e2))
ax.legend()
plt.show()

# PSD
fig, ax = plt.subplots()

bm_freqs, bm_psd = welch(bm_signal, fs=1 / dt, nperseg=1000)
bm_freqs, bm_psd = bm_freqs[1:], bm_psd[1:]  # Remove zero-frequency
bm_freqs = 2 * np.pi * bm_freqs

sp_freqs, sp_psd = welch(signal, fs=1 / dt, nperseg=1000)
sp_freqs, sp_psd = sp_freqs[1:], sp_psd[1:]  # Remove zero-frequency
sp_freqs = 2 * np.pi * sp_freqs

analytical_psd = 4 / (1 + sp_freqs**2)

ax.plot(bm_freqs, bm_psd, color="blue", label="BM")
ax.plot(sp_freqs, sp_psd, color="green", label="SP")
ax.plot(sp_freqs, analytical_psd, color="black", ls="--", label="A")
ax.set_xscale("log")
ax.set_yscale("log")

plt.show()
