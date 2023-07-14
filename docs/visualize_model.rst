.. _visualize-model:


Visualize Model
===============

:py:mod:`blobmodel` provides a ``show_model`` function which makes visualizing a dataset very easy.
``show_model`` creates an animation of the model over time and saves it as a GIF if a filename is provided:


.. code-block:: python

  from blobmodel import show_model

  show_model(ds, gif_name = "example.gif")

You can also adjust the interval between frames in milliseconds with the ``interval`` argument and change the fps of the created GIF with the ``fps`` argument.
A 2D example GIF would look like the following:

.. image:: ../readme_files/2d_blobs.gif
   :alt: StreamPlayer
   :align: center
   :scale: 80%


If you pass a 1D dataset to ``show_model`` the function automatically switches to creating a 1D animation like the following:

.. image:: ../readme_files/1d_blobs.gif
   :alt: StreamPlayer
   :align: center
   :scale: 80%
