# pylint: disable=import-outside-toplevel
"""Generator of Lie closure."""
from collections.abc import Sequence
import logging
from typing import Any, Optional
from fastdla.sparse_pauli_vector import SparsePauliSum, SparsePauliSumArray

LOG = logging.getLogger(__name__)
AlgebraElement = Any
Basis = Sequence[AlgebraElement]


def orthogonalize(
    op: AlgebraElement,
    basis: Basis
) -> AlgebraElement:
    r"""Subtract the subspace projection of an algebra element from itself.

    Let the orthonormal basis be :math:`B = \{g_j\}_{j=0}^{n-1}`. The orthogonal component of
    :math:`h` with respect to :math:`B` is given by

    .. math::

        t = h - \sum_{j=0}^{n-1} \langle g_j, h \rangle g_j.
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

    Lie closure generation follows the standard algorithm of e.g. Algorithm 1 in Wiersema et al. npj
    quant. info. 10 (1):

    Input: Set of generators A
    for a_i in A do
        for a_j in A do
            a_k = [a_i, a_j] / |[a_i, a_j]|
            if a_k not in span(A) then
                A <- A U {a_k}
            endif
        end
    end

    Specific implementations employ different optimization strategies in the loop.

    For an efficient linear independence check, we hold the orthonormalized basis of span(A) in
    memory. The check then becomes equivalent to extracting the normal component of a_k with respect
    to such basis.

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
