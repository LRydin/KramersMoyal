Guidelines to use KramersMoyal
==============================

.. toctree::
   :maxdepth: 2

   license

KramersMoyal
============

Python KM is a python package designed to obtain the Kramers–Moyal
coefficients, or conditional moments, from stochastic data of any
dimension. It employs kernel density estimations, instead of a histogram
approach, to ensure better results for low number of points as well as
allowing better fitting of the results

Installation
============

For the moment the library is available from TestPyPI, so you can use

::

   pip install -i https://test.pypi.org/simple/ kramersmoyal

Then on your favourite editor just use

.. code:: python

   from kramersmoyal import km, kernels

From here you can simply call

.. code:: python

   import numpy as np

   # Number of bins
   bins = np.array([6000])

   # Choose powers to calculate
   powers = np.array([[1], [2]])

   # And here x is your (1D, 2D, 3D) data
   edge, kmc = km(x, bins = bins, powers = powers)


The library depends on ``numpy`` and ``scipy``.


A one-dimensional stochastic process
====================================

The theory
----------

Take for example the well documented one-dimension `Ornstein–Uhlenbeck <https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process>`_
process, also known as Vašíček process.
This process is governed by two main parameters: the mean-reverting
parameter :math:`\theta` and the diffusion parameter :math:`\sigma`


.. math::
   \mathrm{d}y(t) = -\theta y(t)\mathrm{d}t + \sigma \mathrm{d}W(t)

which can be solved in various ways. For our purposes, recall that the drift
coefficient, i.e., the first-order Kramers–Moyal coefficient, is given by
:math:`\mathcal{M}^{[1]}(y) = \theta y` and the second-order Kramers–Moyal
coefficient is :math:`\mathcal{M}^{[2]}(y) = \sigma^2 / 2`, i.e., the diffusion.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
