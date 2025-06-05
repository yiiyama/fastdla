# pylint: disable=import-outside-toplevel
"""Generator of Lie closure."""
from collections.abc import Sequence
import logging
from typing import Any, Optional
from fastdla.sparse_pauli_sum import SparsePauliSum, SparsePauliSumArray

LOG = logging.getLogger(__name__)
AlgebraElement = Any
Basis = Sequence[AlgebraElement]


def orthogonalize(
    op: AlgebraElement,
    basis: Basis
) -> AlgebraElement:
    r"""Subtract the subspace projection of an algebra element from itself.

    Let the orthonormal basis be :math:`V = \{g_j\}_{j=0}^{n-1}`. The orthogonal component of
    :math:`h` with respect to :math:`V` is given by

    .. math::

        h_{\perp} & = h - \mathrm{proj}_V h \\
                  & = h - \sum_{j=0}^{n-1} \langle g_j, h \rangle g_j.

    The inputs to this function can be given in the matrix or SparsePauliSum representations.

    Args:
        op: Operator :math:`h` to be orthogonalized from :math:`V`.
        basis: Basis :math:`V`.

    Returns:
        Orthogonal component :math:`h_{\perp}`.
    """
    if isinstance(op, SparsePauliSum):
        from fastdla._lie_closure_impl.sparse_numba import orthogonalize as fn
    else:
        from fastdla._lie_closure_impl.matrix_jax import orthogonalize as fn

    return fn(op, basis)


def lie_closure(
    generators: Any,
    *,
    keep_original: bool = False,
    max_dim: Optional[int] = None,
    **kwargs
) -> tuple[Basis, Basis] | Basis:
    """Compute the Lie closure of given generators.

    Lie closure generation follows the orthonormalization algorithm in *arXiv:2506.01120*:

    .. code-block:: python

        V = []
        for g in G:
            g_perp = orthogonalize(g, V)
            if g_perp != 0:
                V.append(g_perp / norm(g_perp))

        l = 1
        r = 0
        while l < len(V):
            for m in range(r):
                h = commutator(V[l], V[m])
                h_perp = orthogonalize(h, V)
                if h_perp != 0:
                    V.append(h_perp / norm(h_perp))

            r += 1
            if r == l:
                l += 1
                r = 0

    If keep_original is True, there will be an additional list ``B`` which stores the actual nested
    commutators ``h``. The function then returns both ``B`` and ``V``.

    The inputs to this function can be given in the matrix or SparsePauliSum representations. If
    matrices are given, JAX-based implementation will be called.

    Args:
        generators: Lie algebra elements to compute the closure from.
        keep_original: Whether the returned array of Lie algebra elements should contain the
            original (normalized) generators and their actual nested commutators. If False, the
            orthonormalized basis used internally in the algorithm is returned.
        max_dim: Cutoff for the dimension of the Lie closure. If set, the algorithm may be halted
            before a full closure is obtained.

    Returns:
        A list of linearly independent nested commutators and the orthonormal basis if
        keep_original=True, otherwise only the orthonormal basis.
    """
    if isinstance(generators, list) and isinstance(generators[0], SparsePauliSum):
        generators = SparsePauliSumArray(generators)

    if isinstance(generators, SparsePauliSumArray):
        from fastdla._lie_closure_impl.sparse_numba import lie_closure as fn
    else:
        from fastdla._lie_closure_impl.matrix_jax import lie_closure as fn

    return fn(generators, keep_original=keep_original, max_dim=max_dim, **kwargs)
