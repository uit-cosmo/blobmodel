.. _drainage-time:

Drainage Time
=============

By default, the drainage time is set to a constant value in the whole domain. In this case ``t_drain`` is simply set to an integer or float.
We can also set ``t_drain`` to an array like of length ``Nx``. In this case ``t_drain`` will vary accordingly with x.

Let's take a look at a quick example. Let's assume we want ``t_drain`` to decrease linearly with x. We could implement this as follows:

.. literalinclude:: ../tests/test_docs.py
   :language: python
   :start-after: # PLACEHOLDER drainage_time_0
   :end-before: # PLACEHOLDER drainage_time_1

The time averaged x-profile of ``n`` compared to a constant ``t_drain`` = 2 would then look like this:

.. image:: change_t_drain_plot.png
   :scale: 80%
