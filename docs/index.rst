.. blobmodel documentation master file, created by
   sphinx-quickstart on Fri Jun 16 13:31:30 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: logo.png
   :alt: PlasmaPy logo
   :align: right
   :scale: 40%


blobmodel Documentation
=======================

:py:mod:`blobmodel` is a python package for creating numerical realizations of advecting and dissipating blobs in up to two spatial dimensions.
This open source project is developed by the Complex Systems Modelling group at UiT The Arctic University of Norway. 

:py:mod:`blobmodel` is developed openly `on GitHub <https://github.com/uit-cosmo/blobmodel>`_, where you can request a new feature, report a bug or contribute to the package by creating a pull request.

Mathematically, the model of :math:`K` superposed blobs, :math:`\Phi_K`, can be expressed as 

.. math::

   \Phi_K(x, y, t) = \sum_{k=1}^{K} \phi_k(x, y, t).

The blobs :math:`\phi_k` consists of an amplitude :math:`a_k` and a blob shape :math:`\varphi` with a blob width in x and y, given by :math:`l_x` and :math:`l_y`: 

.. math::

   \phi(x, y, t) = a_k \varphi\left(\frac{x}{l_x}, \frac{y}{l_y}, t \right). 

Each blob has an individual velocity in x and y, :math:`v_x` and :math:`v_y`, and decreases in amplitude with the drainage time :math:`\tau`:

.. math::
   
  \frac{\partial \phi_k}{\partial t} + v_x \frac{\partial \phi_k}{\partial x} + v_y \frac{\partial \phi_k}{\partial y} + \frac{\phi_k}{\tau} = 0.
   

A 2D example of the model is shown below:

.. image:: ../readme_files/2d_blobs.gif
   :alt: StreamPlayer
   :align: center
   :scale: 80%

The model can also be reduced to one spatial dimension as shown in the following example:

.. image:: ../readme_files/1d_blobs.gif
   :alt: StreamPlayer
   :align: center
   :scale: 80%

The following sections provide detailed information about the package components.
Alternatively, you can take a look at the examples gallery at :py:mod:`blobmodel/examples/` for a quick overview of the package's functionalities.


.. toctree::
   :caption: Contents
   :maxdepth: 1

   Installing <install>
   getting_started
   visualize_model
   model_geometry
   blob_shapes
   one_dim
   blob_factory
   blob_labels
   drainage_time
   blob_tilt
   contributor_guide


.. toctree::
   :maxdepth: 2
   :caption: API reference

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
