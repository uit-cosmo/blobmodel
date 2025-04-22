.. _getting-started:
   
Getting Started
===============

++++++++++++++++
Creating a model
++++++++++++++++

We create a grid on which the blobs are discretized with using the ``Model`` class. Here, we specify the geometry of the model by the number of grid points, the lengths of the domain, the time step, the time length and the initial time of our realization.

In addition, we can specify the blob shape, drainage time and the number of blobs when creating a ``Model`` object. For more details about the geometry, take a look at the :ref:`blobmodel-geometry` section.

.. literalinclude:: ../tests/test_docs.py
  :language: python

.. code-block:: python

  from blobmodel import Model

    bm = Model(
        Nx=100,
        Ny=100,
        Lx=10,
        Ly=10,
        dt=0.1,
        T=20,
        periodic_y=True,
        blob_shape=BlobShapeImpl(BlobShapeEnum.gaussian),
        num_blobs=100,
        t_drain=1e10,
        t_init=10,
    )



+++++++++++++++++
Superposing blobs
+++++++++++++++++

We can now call the ``make_realization()`` method to sum up the individual blobs. The blob parameters are sampled from the according distribution functions (see :ref:`parameter-distributions` for further details).
If we provide a ``file_name`` to the ``make_realization`` method, it will store the realization under this name on your machine. 

.. code-block:: python 

   ds = bm.make_realization(file_name="example.nc")

The ``make_realization`` mehtod can take two more arguments, ``speed_up`` and ``error``, which can be helpful for integrating very large datasets. 
By setting ``spee_up`` to ``True``, the code will truncate the blobs when the blob values fall under the given ``error`` value. 
The code assumes an exponential shape for the blobs when calculating the truncation position (see :ref:`blob-shapes` for further details).
The resulting realization will look something like this:


.. image:: 2d_animation.gif
   :alt: StreamPlayer
   :align: center
   :scale: 80%

Which may be a bit boring, as all the blobs move with the same speed and have the same shape and size. We will learn how to set distribution functions for the blobs parameters in :ref:`parameter-distributions`


The ``make_realization`` method returns the realization in the form of an `xarray dataset <https://docs.xarray.dev/en/stable/index.html>`_. 
The superposed pulses are stored in the ``n`` variable of the dataset. We can now analyse the data using the convenient xarray syntax, for example:

.. code-block:: python

  import matplotlib.pyplot as plt

  ds["n"].isel(y=0).mean(dim=("t")).plot()
  plt.show()

.. image:: xarray_example.png
   :scale: 80%
