import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(-5, 5, 1000)

exp = np.exp(t) * np.heaviside(-1.0 * t, 1)
lorentz = 1 / (np.pi * (1 + t**2))
gauss = 1 / np.sqrt(2 * np.pi) * np.exp(-(t**2) / 2)
secant = 2 / np.pi / (np.exp(t) + np.exp(-t))
dipole = -2 * t / np.sqrt(np.pi) * np.exp(-(t**2))

plt.plot(t, exp, label="exp")
plt.plot(t, lorentz, label="lorentz")
plt.plot(t, gauss, label="gauss")
plt.plot(t, secant, label="secant")
plt.plot(t, dipole, label="dipole")
plt.legend()
plt.xlabel("t")
plt.ylabel("n")
plt.savefig("pulse_shapes.png")
plt.show()

for lam in [0.2, 0.5, 0.8]:
    shape = np.zeros(1000)
    shape[t < 0] = np.exp(t[t < 0] / lam)
    shape[t >= 0] = np.exp(-t[t >= 0] / (1 - lam))
    plt.plot(t, shape, label=f"lam = {lam}")

plt.xlabel("t")
plt.ylabel("n")
plt.legend()
plt.savefig("2-sided_pulse_shape.png")
plt.show()
