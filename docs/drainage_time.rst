.. _drainage-time:

Drainage Time
=============

By default, the drainage time is set to a constant value in the whole domain. In this case ``t_drain`` is simply set to an integer or float.
We can also set ``t_drain`` to an array like of length ``Nx``. In this case ``t_drain`` will vary accordingly with x.

Let's take a look at a quick example. Let's assume we want ``t_drain`` to decrease linearly with x. We could implement this as follows:

.. code-block:: python

  from blobmodel import Model, DefaultBlobFactory
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

  ds_changing_t_drain = tmp.make_realization()

The time averaged x-profile of ``n`` compared to a constant ``t_drain`` = 2 would then look like this: 

.. image:: change_t_drain_plot.png
   :scale: 80%
