.. _one_dim:
   
One Dimensional
===========

The default behaviour of the ``Model`` is to draw two-dimensional realizations. It is possible to draw one-dimensional realizations too.
This is achieved by setting the flag ``one_dimensional`` to True. Additionally, the vertical velocities should be set to zero:

.. literalinclude:: ../tests/test_docs.py
   :language: python
   :start-after: # PLACEHOLDER one_dim_0
   :end-before: # PLACEHOLDER one_dim_1

Such a model will result in the following realization:

.. code-block:: python

    show_model(dataset=ds, interval=100, gif_name="1d_animation.gif")


.. image:: 1d_animation.gif
   :alt: StreamPlayer
   :align: center
   :scale: 80%
