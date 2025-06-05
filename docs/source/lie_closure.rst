====================================================
Construction of Lie closure from a set of generators
====================================================

The main function of fastdla, ``lie_closure``, generates the closure of the given matrices by
iteratively evaluating all nested commutators. The returned value is an array representing the
*orthonormal basis* of the subalgebra.

To compute the closure, you first have to define the generators, i.e., the seed matrices of which
the nested commutators will be evaluated. The generators can be constructed "by hand" as numpy
arrays, or (for e.g. quantum computing use cases) defined as sums of Pauli strings using the
built-in ``SparsePauliSum`` class. If the full ("dense") matrix representation of the generators
would not fit in the memory but the sparse representation would, you can pass the ``SparsePauliSum``
objects directly to ``lie_closure``. Otherwise, you would want to convert the sparse generators to
numpy arrays using the conversion method of ``SparsePauliSum``, since matrix calculations tend to
be faster (especially if you are running on a GPU).

Some example functions to define the generators are provided in ``fastdla.generators``. Currently
we have implementations for several "ansatz circuits" used in variational quantum algorithms:

* Hardware-efficient ansatz [1]
* Spin glass Hamiltonian variational ansatz (HVA) [1]
* Heisenberg model HVA [1]
* Transverse field Ising model HVA [1]
* Z2 lattice gauge theory HVA [2]

Sources:

| [1] Larocca, et al. Quantum 6, 824 (2022).
| [2] Nagano et al. To be published.
