.. _blob-shapes:
   

Blob Shapes
===========

We can choose between four different blob shapes that are specified with the ``blob_shape`` argument of the ``Model`` class.

The blob shape consists of two parts, the blob shape in the propagation direction and the blob shape in the perpendicular direction thereof.
The propagation direction is calculated from vx and vy of each individual blob (see :ref:`blob-alignment` for further details).

The blob shape in the perpendicular direction is always set to "gauss", whereas you can choose one of the following blob shapes for the propagation direction:

.. list-table:: 
   :widths: 10 10 10 10
   :header-rows: 1

   * - "exp"
     - "gauss"
     - "lorentz"
     - "secant"
   * - Exponential Pulse
     - Gaussian Pulse
     - Lorentz Pulse
     - Secant Pulse
   * - ``np.exp(theta) * np.heaviside(-1.0 * t, 1)``
     - ``1 / np.sqrt(np.pi) * np.exp(-(t**2))``
     - ``1 / (np.pi * (1 + t**2))``
     - ``2 / np.pi / (np.exp(t) + np.exp(-t))``

.. image:: pulse_shapes.png
   :scale: 80%
