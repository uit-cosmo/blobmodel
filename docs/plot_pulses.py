import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(-5, 5, 1000)

exp = np.exp(t) * np.heaviside(-1.0 * t, 1)
lorentz = 1 / (np.pi * (1 + t**2))
gauss = 1 / np.sqrt(np.pi) * np.exp(-(t**2))
secant = 2 / np.pi / (np.exp(t) + np.exp(-t))

plt.plot(t, exp, label="exp")
plt.plot(t, lorentz, label="lorentz")
plt.plot(t, gauss, label="gauss")
plt.plot(t, secant, label="secant")
plt.legend()
plt.xlabel("t")
plt.ylabel("n")
plt.savefig("pulse_shapes.png")
plt.show()
