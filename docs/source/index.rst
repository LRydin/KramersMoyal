Kramers---Moyal
===============

.. toctree::
   :maxdepth: 2

:code:`kramersmoyal` is a python package designed to obtain the Kramers---Moyal
coefficients, or conditional moments, from stochastic data of any
dimension. It employs kernel density estimations, instead of a histogram
approach, to ensure better results for low number of points as well as
allowing better fitting of the results.

.. include:: installation.rst

.. include:: 1dprocess.rst

.. include:: 2dprocess.rst


Table of Content
================

.. toctree::
   :maxdepth: 3

   installation
   1dprocess
   2dprocess
   functions/index
   license

Literature
==========

:sup:`1` Friedrich, R., Peinke, J., Sahimi, M., Tabar, M. R. R. *Approaching complexity by stochastic methods: From biological systems to turbulence,* [Phys. Rep. 506, 87–162 (2011)](https://doi.org/10.1016/j.physrep.2011.05.003).

The study of stochastic processes from a data-driven approach is grounded in extensive mathematical work. From the applied perspective there are several references to understand stochastic processes, the Fokker---Planck equations, and the Kramers---Moyal expansion

| Tabar, M. R. R. (2019). *Analysis and Data-Based Reconstruction of Complex Nonlinear Dynamical Systems.* Springer, International Publishing
| Risken, H. (1989). *The Fokker–Planck equation.* Springer, Berlin, Heidelberg.
| Gardiner, C.W. (1985). *Handbook of Stochastic Methods.* Springer, Berlin.

An extensive review on the subject can be found `here <http://sharif.edu/~rahimitabar/pdfs/80.pdf>`_.

Funding
=======

Helmholtz Association Initiative *Energy System 2050 - A Contribution of the Research Field Energy* and the grant No. VH-NG-1025 and *STORM - Stochastics for Time-Space Risk Models* project of the Research Council of Norway (RCN) No. 274410.
