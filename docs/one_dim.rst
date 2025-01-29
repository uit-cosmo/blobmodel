.. _one_dim:
   
One Dimensional
===========

The default behaviour of the ``Model`` is to draw two-dimensional realizations. It is possible to draw one-dimensional realizations too.
This is achieved by setting the flag ``one_dimensional`` to True. Additionally, the vertical velocities should be set to zero:

.. code-block:: python

    from blobmodel import Model, show_model, DefaultBlobFactory

    bf = DefaultBlobFactory(A_dist="exp", wx_dist="deg", vx_dist="deg", vy_dist="zeros")

    bm = Model(
        Nx=100,
        Ny=1,
        Lx=10,
        Ly=0,
        dt=0.1,
        T=10,
        periodic_y=False,
        blob_shape="exp",
        num_blobs=20,
        t_drain=10,
        blob_factory=bf,
        one_dimensional=True,
    )

Such a model will result in the following realization:

.. code-block:: python

    ds = bm.make_realization(speed_up=True, error=1e-2)
    show_model(dataset=ds, interval=100, gif_name="1d_animation.gif")


.. image:: 1d_animation.gif
   :alt: StreamPlayer
   :align: center
   :scale: 80%
