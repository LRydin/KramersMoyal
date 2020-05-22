Installation
============

To install :code:`kramersmoyal`, just use `pip`

::

   pip install kramersmoyal

Then on your favourite editor just use

.. code:: python

   from kramersmoyal import km

From here you can simply call

.. code:: python

   import numpy as np

   # Number of bins
   bins = np.array([6000])

   # Choose powers to calculate
   powers = np.array([[1], [2]])

   # And here x is your (1D, 2D, 3D) data
   kmc, edge = km(x, bins = bins, powers = powers)

The library depends on :code:`numpy` and :code:`scipy`.
