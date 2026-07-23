.. blob-factory:

Parameter Distributions
=======================

By default a realization obtained by an instance of ``Model`` will have degenerate distributions of all blob parameters. For any interesting application this is not enough.
In order to provide the highest possible flexibility for setting blob parameters this is done by a separate class, which
should implement ``BlobFactory``. An implementation is provided by ``DefaultBlobFactory`` which allows to easily set independent distributions of any blob parameters to a wide range of distribution families.

++++++++++++++++++
DefaultBlobFactory
++++++++++++++++++

We can use ``DefaultBlobFactory`` class in order to change the distribution functions of the following blob parameters:

* Amplitudes
* widths in x and y
* velocities in x and y
* pulse shape in x and y
* blob alignment and tilt

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

.. literalinclude:: ../tests/test_docs.py
   :language: python
   :start-after: # PLACEHOLDER blob_factory_0
   :end-before: # PLACEHOLDER blob_factory_1

++++++++++++++++++++++++++++++++++
Pre-built blobs: Model.from_blobs
++++++++++++++++++++++++++++++++++

If you already know exactly which blobs you want to realize, you don't need parameter distributions at all:
build the ``Blob`` objects yourself and hand the list to ``Model.from_blobs``.
Each ``Blob`` carries its own parameters, so the sampling arguments of ``Model``
(``num_blobs`` and ``blob_shape``) are not needed:

.. literalinclude:: ../tests/test_docs.py
   :language: python
   :start-after: # PLACEHOLDER prebuilt_blobs_0
   :end-before: # PLACEHOLDER prebuilt_blobs_1

``Model.from_blobs`` is a shortcut for wrapping the list in a ``BlobListFactory``, which you can also use directly with ``Model(blob_factory=...)``.
Note that ``t_drain=np.inf`` is the documented way of specifying non-draining blobs.

+++++++++++++++++++
CallableBlobFactory
+++++++++++++++++++

If you want custom randomness per blob without writing a whole ``BlobFactory`` subclass, use ``CallableBlobFactory``:
it builds each blob by calling a getter function with the factory's random number generator.
Drawing the random numbers from that generator (instead of the global ``np.random`` state) makes the realization reproducible via ``CallableBlobFactory(seed=...)`` or ``Model(seed=...)``:

.. literalinclude:: ../tests/test_docs.py
   :language: python
   :start-after: # PLACEHOLDER callable_blob_factory_0
   :end-before: # PLACEHOLDER callable_blob_factory_1

+++++++++++++++++
CustomBlobFactory
+++++++++++++++++

But what if you want to use a distribution function that is not implemented? Or maybe you even want to change the initial position and arrival time of each blob?
In this case you can use the ``CustomBlobFactory`` class to define all blob parameters individually. An example could look like this:

.. literalinclude:: ../tests/test_docs.py
   :language: python
   :start-after: # PLACEHOLDER custom_blob_factory_0
   :end-before: # PLACEHOLDER custom_blob_factory_1

By assigning an array like to the variables ``amp``, ``width``, ``vx``, ``vy``, ``posx``, ``posy`` and ``t_init`` we can exactly define every single blob parameter of every single blob.

.. note::

   When using ``CustomBlobFactory`` it is your responsibility to make sure all blob variables have the correct dimensions. Also, if you wish to normalize the parameters you have to do this manually.
