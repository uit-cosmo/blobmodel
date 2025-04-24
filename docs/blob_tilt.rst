.. _blob-tilt:

Blob tilt
==============

The code for generating the realizations presented in this file is located in examples/blob_tilting.py

By default, the blob shape is rotated to its propagation direction. This means that the rotation angle
:math:`\theta` is calculated as ``cmath.phase(vx + vy * 1j)``. This behaviour is set when the blob_alignment
attribute in the Blob class is set to True, which is the default.

An example of a realization with aligned blobs is shown below.

.. image:: alignment_true.gif
   :alt: StreamPlayer
   :align: center
   :scale: 80%

Alternatively, we can force force any tilt angle by setting blob_alignment = False for each blob. If you use a ``DefaultBlobFactory``, this can be done by setting a False ``blob_alignment`` flag:

.. literalinclude:: ../tests/test_docs.py
   :language: python
   :start-after: # PLACEHOLDER blob_tilt_0
   :end-before: # PLACEHOLDER blob_tilt_1

Then the tilt will be given by the argument theta. Which, if using a ``DefaultBlobFactory``, can be set by

.. literalinclude:: ../tests/test_docs.py
   :language: python
   :start-after: # PLACEHOLDER blob_tilt_1
   :end-before: # PLACEHOLDER blob_tilt_2

Setting the angle with a lambda allows us to set a distribution of tilt angles. In this case we use a degenerate distribution:
The blob propagation direction won't be affected. The resulting realization is shown below:

.. image:: alignment_false.gif
   :alt: StreamPlayer
   :align: center
   :scale: 80%
