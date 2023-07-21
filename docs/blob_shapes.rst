.. _blob-shapes:
   

Blob Shapes
===========

We can choose between four different blob shapes that are specified with the ``blob_shape`` argument of the ``Model`` class.

The blob shape consists of two parts, the blob shape in the propagation direction and the blob shape in the perpendicular direction thereof.
The propagation direction is calculated from vx and vy of each individual blob (see :ref:`blob-alignment` for further details).

You can choose one of the following blob shapes for the propagation and perpendicular direction:

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

+++++++++++++++++++++
Propagation direction
+++++++++++++++++++++

If you specify the ``blob_shape`` as a string such as ``blob_shape = "exp"``, the specified blob shape will be used in the propagation direction. 
The perpendicular blob shape will be set to ``gauss``.

+++++++++++++++++++++++
Perpendicular direction
+++++++++++++++++++++++

In order to specify the blob shape in the perpendicular direction we need to specify the blob shapes with the ``BlobShapeImpl`` class.
The first argument refers to the propagation direction and the second one refers to the perpendicular direction.
An example would look like this:

.. code-block:: python

  from blobmodel import Model, BlobShapeImpl

  bm = Model(
      Nx=100,
      Ny=100,
      Lx=10,
      Ly=10,
      dt=0.1,
      T=10,
      num_blobs=10,
      blob_shape=BlobShapeImpl("exp", "lorentz"),
      periodic_y=True,
      t_drain=1e10,
  )


++++++++++++++++++++++++++++++++
Two-sided exponential blob shape
++++++++++++++++++++++++++++++++

The last blob shape we discuss is the two-sided exponential blob shape. In contrast to the shapes above, it requires an asymmetry parameter ``lam`` to specify the exact shape.
The shape is implemented as follows:

.. code-block:: python

  shape[t < 0] = np.exp(t[t < 0] / lam)
  shape[t >= 0] = np.exp(-t[t >= 0] / (1 - lam))

.. image:: 2-sided_pulse_shape.png
   :scale: 80%

We specify the asymmetry parameter when defining the ``blob_factory``. An example would look like this:

.. code-block:: python

  bf = DefaultBlobFactory(
      A_dist="deg",
      wx_dist="deg",
      spx_dist="deg",
      spy_dist="deg",
      shape_param_x_parameter=0.5,
      shape_param_y_parameter=0.5,
  )

  bm = Model(
      Nx=100,
      Ny=100,
      Lx=10,
      Ly=10,
      dt=0.1,
      T=10,
      num_blobs=10,
      blob_shape=BlobShapeImpl("2-exp", "2-exp"),
      t_drain=1e10,
      blob_factory=bf,
  )

Take a look at ``examples/2_sided_exp_pulse.py`` for a fully implemented example.
