.. _parameter-distributions:

Parameter Distributions
=======================

While instantiating a ``Model`` object, we define the distribution functions used to sample blob parameters from. 
So far we have only used the default distributions but in this section we cover how change these.

++++++++++++++++++
DefaultBlobFactory
++++++++++++++++++

We can use ``DefaultBlobFactory`` class in order to change the distribution functions of the following blob parameters:

* Amplitudes
* widths in x
* widths in y
* velocities in x
* velocities in y

The distributions for these parameters are set with the ``*_dist`` arguments. The following distribution functions are implemented:

.. list-table:: 
   :widths: 10 50
   :header-rows: 0

   * - "exp"
     - exponential distribution with mean 1

   * - "gamma"
     - gamma distribution with ``free_parameter`` as shape parameter and mean 1

   * - "normal"
     - normal distribution with zero mean and ``free_parameter`` as scale parameter

   * - "uniform"
     - uniform distribution with mean 1 and ``free_parameter`` as width

   * - "ray"
     - Rayleight distribution with mean 1

   * - "deg"
     - degenerate distribution at ``free_parameter``

   * - "zeros"
     - array of zeros

As you can see, some of these distributions require an additional ``free_parameter`` to specify the distribution.
This is done by the ``*_parameter`` arguments.

Let's take a look at an example. Let's assume we want to sample the blob amplitudes from normal distribution with 5 as the scale parameter. 
We chose the defualt distributions for all other blob parameters. We would set up the Model as follows:


.. code-block:: python

  from blobmodel import Model, DefaultBlobFactory

  my_blob_factory = DefaultBlobFactory(A_dist="normal", A_parameter=5)

  bm = Model(
      Nx=200,
      Ny=100,
      Lx=10,
      Ly=10,
      dt=0.1,
      T=20,
      blob_shape="gauss",
      blob_factory=my_blob_factory,
      t_drain=100,
      num_blobs=100,
  )

  ds = bm.make_realization()

+++++++++++++++++
CustomBlobFactory
+++++++++++++++++

But what if you want to use a distribution function that is not implemented? Or maybe you even want to change the initial position and arrival time of each blob?
In this case you can use the ``CustomBlobFactory`` class to define all blob parameters individually. An example could look like this:

.. code-block:: python

   from blobmodel import (
    Model,
    BlobFactory,
    Blob,
    AbstractBlobShape,
    )
    import numpy as np

    # create custom class that inherits from BlobFactory
    # here you can define your custom parameter distributions
    class CustomBlobFactory(BlobFactory):
        def __init__(self) -> None:
            pass

        def sample_blobs(
            self,
            Ly: float,
            T: float,
            num_blobs: int,
            blob_shape: AbstractBlobShape,
            t_drain: float,
        ) -> list[Blob]:

            # set custom parameter distributions
            amp = np.linspace(0.01, 1, num=num_blobs)
            width = np.linspace(0.01, 1, num=num_blobs)
            vx = np.linspace(0.01, 1, num=num_blobs)
            vy = np.linspace(0.01, 1, num=num_blobs)

            posx = np.zeros(num_blobs)
            posy = np.random.uniform(low=0.0, high=Ly, size=num_blobs)
            t_init = np.random.uniform(low=0, high=T, size=num_blobs)

            # sort blobs by _t_init
            t_init = np.sort(t_init)

            return [
                Blob(
                    blob_id=i,
                    blob_shape=blob_shape,
                    amplitude=amp[i],
                    width_prop=width[i],
                    width_perp=width[i],
                    v_x=vx[i],
                    v_y=vy[i],
                    pos_x=posx[i],
                    pos_y=posy[i],
                    t_init=t_init[i],
                    t_drain=t_drain,
                )
                for i in range(num_blobs)
            ]

        def is_one_dimensional(self) -> bool:
            return False


    bf = CustomBlobFactory()
    tmp = Model(
        Nx=100,
        Ny=100,
        Lx=2,
        Ly=2,
        dt=0.1,
        T=10,
        blob_shape="gauss",
        t_drain=2,
        periodic_y=True,
        num_blobs=1000,
        blob_factory=bf,
    )

    ds = tmp.make_realization()

By assigning an array like to the variables ``amp``, ``width``, ``vx``, ``vy``, ``posx``, ``posy`` and ``t_init`` we can exactly define every single blob parameter of every single blob.

.. note::

   When using ``CustomBlobFactory`` it is your responsibility to make sure all blob variables have the correct dimensions. Also, if you wish to normalize the parameters you have to do this manually.
