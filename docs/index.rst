.. QUARK documentation master file, created by
   sphinx-quickstart on Fri Feb  4 11:04:21 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to QUARK's documentation!
=================================

**QUantum computing Application benchmaRK** (QUARK) is a framework for designing, implementing, executing, and analyzing
benchmarks. The QUARK framework aims to facilitate the development of application-level benchmarks. The framework simplifies
the end-to-end process of designing, implementing, conducting, and communicating application benchmarks.

**Note:** This documentation is currently being built and is not complete!

The following figure depicts the main components of the framework. The framework
follows the separation of concerns design principle encapsulating application- and problem-specific aspects, mappings
to mathematical formulations, solvers, hardware or other custom module definitions.

.. image:: architecture.png
  :align: center
  :width: 700
  :alt: Architecture of the QUARK framework

Paper
======

Details about the motivations for the framework can be seen in the accompanying QUARK paper (`arXiv link
<https://arxiv.org/abs/2202.03028>`_). Even though the architecture changed quite significantly with the 2.0 release of QUARK, the guiding principles still remain.
The data used for the original paper can be found in ``paper/results.csv``.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorial
   developer
   analysis
   reference


License
========

QUARK is licensed under Apache 2.0 license.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
