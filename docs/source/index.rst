.. qudit-sim documentation master file, created by
   sphinx-quickstart on Tue Apr 12 17:15:26 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :hidden:

   Home <self>
   Lie closure <lie_closure>
   Eigenspace projection <eigenspace>

.. toctree::
   :hidden:
   :caption: API Reference

   apidocs/lie_closure
   apidocs/eigenspace
   apidocs/pauli
   apidocs/generators

=====================================
Welcome to fastdla's documentation!
=====================================

``fastdla`` is a library for numerical evaluation of Lie closures. A Lie closure is the smallest
subalgebra of a Lie algebra that contains a given set of algebra elements. Lie closures of the
generators of parametric quantum circuits are also known as dynamical Lie algebras (DLAs), whose
properties such as the dimensionality are known to contain rich information about the associated
circuits.

In general, generating the :math:`N`-dimensional closure involves evaluations of at least
:math:`N (N-1) / 2` Lie brackets (matrix commutators) and a check of linear independence of the
result of each Lie bracket calculation with respect to the other elements in the closure. While
there is little room to accelerate Lie bracket calculations, determination of linear independence
can be done in multiple ways with different computational complexities.
`ArXiv:2505.01120 <https://arxiv.org/abs/2506.01120>`_ proposed a few algorithms that scaled
preferrably compared to a well-known method using the singular value decomposition. ``fastdla``
gives implementations of the proposed algorithms using `Numba <https://numba.pydata.org>`_ and
`JAX <https://docs.jax.dev>`_.


Project status
==============

The main algorithms are implemented and believed to be stable. New generators may be added.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
