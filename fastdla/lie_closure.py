# pylint: disable=import-outside-toplevel
"""Generator of Lie closure."""
from collections.abc import Sequence
import logging
from typing import Any, Optional
from fastdla.sparse_pauli_vector import SparsePauliVector, SparsePauliVectorArray

LOG = logging.getLogger(__name__)
AlgebraElement = Any
Basis = Sequence[AlgebraElement]
InnerProductMatrix = Any


def linear_independence(
    op: Any,
    basis: Basis,
    xinv: Optional[InnerProductMatrix] = None
) -> bool:
    r"""Check if the given operator is linearly independent from all other elements in the basis.

    Let the Lie algebra elements in the basis :math:`B` be :math:`g_0, g_1, ..., g_{n-1}`. If a new
    element :math:`h` is linearly dependent on :math:`\{g_j\}_{j=0}^{n-1}`, there is a column vector
    of coefficients :math:`\mathbf{a} = (a_0, a_1, ..., a_{n-1})^T` where

    .. math::

        \sum_{k=0}^{n-1} g_k a_k = h.

    Using the inner product between algebra elements, let :math:`X_{jk} = \langle g_j, g_k \rangle`.
    Inner product of both sides with :math:`g_j` yields

    .. math::

        \sum_{k=0}^{n-1} X_{jk} a_k = \langle g_j, h \rangle.

    Since :math:`B` is linearly independent, the matrix :math:`X` is invertible. Therefore

    .. math::

        a_j = \sum_{k=0}^{n-1} (X^{-1})_{jk} \langle g_k, h \rangle.

    Using thus calculated :math:`\{a_j\}_{j=0}^{n-1}`, we check the residual

    .. math::

        \Delta = h - \sum_{j=0}^{n-1} g_j a_j

    to determine the linear independence of :math:`h` with respect to :math:`B`.

    Args:
        new_op: Lie algebra element :math:`h` to check the linear independence of.
        basis: The basis (list of linearly independent elements) of the Lie Algebra.
        xinv: Inverse of the :math:`X` matrix. If not given, computed from basis.

    Returns:
        True if :math:`h` is linearly independent from all elements of the basis.
    """
    if isinstance(op, SparsePauliVector):
        from fastdla._lie_closure_impl.sparse_numba import linear_independence as fn
    else:
        from fastdla._lie_closure_impl.matrix_jax import linear_independence as fn

    return fn(op, basis, xinv)


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
    if isinstance(op, SparsePauliVector):
        from fastdla._lie_closure_impl.sparse_numba import orthogonalize as fn
    else:
        from fastdla._lie_closure_impl.matrix_jax import orthogonalize as fn

    return fn(op, basis)


def lie_closure(
    generators: Any,
    *,
    keep_original: bool = True,
    max_dim: Optional[int] = None,
    **kwargs
) -> Basis:
    """Compute the Lie closure of given generators.

    Lie closure generation follows the standard algorithm of e.g. Algorithm 1 in Wiersema et al. npj
    quant. info. 10 (1):

    Input: Set of generators A
    for a_i in A do
        for a_j in A do
            a_k = [a_i, a_j]
            if a_k not in span(A) then
                A <- A U {a_k}
            endif
        end
    end

    Specific implementations employ different optimization strategies in the loop implementations.

    The algorithm to determine the linear independence of a_k with A is described in
    `linear_independence`. When we are allowed to modify the original generators so that A can be
    orthonormalized, the linear independence can be determined through checking the existence of an
    orthogonal component of a_k with respect to the subspace spanned by A. If such a component
    exists, its normalized form is added to A in the update step.

    Args:
        generators: Lie algebra elements to compute the closure from.
        keep_original: Whether to keep the original generator elements. If False, only
            orthonormalized Lie algebra elements are kept in memory to speed up the calculation.
        max_dim: Cutoff for the dimension of the Lie closure. If set, the algorithm may be halted
            before a full closure is obtained.
        min_tasks: (CPU-based implementation only) Minimum number of commutator calculations to
            complete before starting a new batch of calculations.
        max_workers: (CPU-based implementation only) Maximun number of threads to use to parallelize
            the commutator calculations.

    Returns:
        A basis of the Lie closure.
    """
    if isinstance(generators, list) and isinstance(generators[0], SparsePauliVector):
        generators = SparsePauliVectorArray(generators)

    if isinstance(generators, SparsePauliVectorArray):
        from fastdla._lie_closure_impl.sparse_numba import lie_closure as fn
    else:
        from fastdla._lie_closure_impl.matrix_jax import lie_closure as fn

    return fn(generators, keep_original=keep_original, max_dim=max_dim, **kwargs)
