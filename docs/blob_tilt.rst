.. _blob-alignment:

Blob Alignment
==============

By default, the blob shape is rotated to its propagation direction. The rotation angle :math:`\theta` is calculated as ``cmath.phase(vx + vy * 1j)``.

An example of a rotated blob is shown below.

.. image:: alignment_true.gif
   :alt: StreamPlayer
   :align: center
   :scale: 80%

Alternatively, we can force force any tilt angle by setting blob_alignment = False for each blob. Then the tilt will be
given by the argument theta. The blob propagation direction won't be affected.

.. image:: alignment_false.gif
   :alt: StreamPlayer
   :align: center
   :scale: 80%

