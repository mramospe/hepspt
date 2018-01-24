=======
hep_spt
=======

The High Energy Physics Statistics and Plotting Tools package provides tools to work in High Energy Physics using general python packages.

Main points
===========

  * Statistical functions to work Bayesian/Frequentist approaches.
  * Utilities to handle poissonian/weighted histograms.
  * Tools to handle figures with `matplotlib <https://matplotlib.org/>`_.

Considerations:
===============

  * Samples are preferred as structured numpy.ndarray or pandas.DataFrame objects.
  * Plotting functions and classes are designed to work with matplotlib.
  * Statistical tools are built on top of the standard scipy package.

Installation:
=============

To use the **latest development version**, clone and install with `pip`:

.. code-block:: bash

   git clone https://github.com/mramospe/hep_spt.git
   cd hep_spt
   sudo pip install .
