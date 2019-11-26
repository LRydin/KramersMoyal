KramersMoyal's documentation!
=============================


Header
======


Guide
^^^^^

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

Dependencies
------------

The library depends on ``numpy`` and ``scipy``.


A one-dimensional stochastic process
====================================

The theory
----------

Take for example the well documented one-dimension `Ornstein–Uhlenbeck <https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process>`_
process, also known as Vašíček process.
This process is governed by two main parameters: the mean-reverting
parameter θ and the diffusion parameter σ


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
