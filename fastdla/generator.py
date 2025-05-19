"""Generator of Lie closure."""
from collections.abc import Sequence
from typing import Any, Optional
from fastdla.sparse_pauli_vector import SparsePauliVector, SparsePauliVectorArray
import fastdla._generator_impl.sparse as impl_sparse
import fastdla._generator_impl.matrix as impl_matrix

AlgebraElement = Any
Basis = Sequence[AlgebraElement]
InnerProductMatrix = Any


def linear_independence(
    op: Any,
    basis: Basis,
    xinv: Optional[InnerProductMatrix] = None
) -> bool:
    """Check if the given operator is linearly independent from all other elements in the basis.

    Let the Lie algebra elements in the basis be P0, P1, ..., Pn. The basis_matrix Π is a matrix
    formed by stacking the column vectors {Pi}:
    Π = (P0, P1, ..., Pn).
    If a new element Q is linearly dependent on {Pi}, there is a column vector of coefficients
    a = (a0, a1, ..., an)^T where
    Π a = Q.
    Multiply both sides with Π† and denote X = Π†Π to obtain
    X a = Π† Q.
    Since {Pi} are linearly independent, X must be invertible:
    a = X^{-1} Π† Q.
    Using thus calculated {ai}, we check the residual
    R = Q - Π a
    to determine the linear independence of Q with respect to {Pi}.

    Args:
        new_op: Lie algebra element Q to check the linear independence of.
        basis: The basis (list of linearly independent elements) of the Lie Algebra.
        xinv: Inverse of the X matrix. If not given, computed from basis.

    Returns:
        True if Q is linearly independent from all elements of the basis.
    """
    if isinstance(op, SparsePauliVector):
        return impl_sparse.linear_independence(op, basis, xinv)
    else:
        return impl_matrix.linear_independence(op, basis, xinv)


def orthogonalize(
    op: AlgebraElement,
    basis: Basis
) -> AlgebraElement:
    """Subtract the subspace projection of an algebra element from itself.

    Let the orthonormal basis be P0, P1, ..., Pn. The basis_matrix Π is a matrix formed by stacking
    the column vectors {Pi}:
    Π = (P0, P1, ..., Pn).
    The orthogonal component of Q with respect to the basis is given by
    T = Q - Π Π† Q.
    """
    if isinstance(op, SparsePauliVector):
        return impl_sparse.orthogonalize(op, basis)
    else:
        return impl_matrix.orthogonalize(op, basis)


def lie_closure(
    generators: Any,
    *,
    keep_original: bool = True,
    max_dim: Optional[int] = None,
    verbosity: int = 0,
    min_tasks: int = 0,
    max_workers: Optional[int] = None
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
    linear_independence(). When we are allowed to modify the original generators so that A can be
    orthonormalized, the linear independence can be determined through checking the existence of an
    orthogonal component of a_k with respect to the subspace spanned by A. If such a component
    exists, its normalized form is added to A in the update step.

    Args:
        generators: Lie algebra elements to compute the closure from.
        keep_original: Whether to keep the original generator elements. If False, only
            orthonormalized Lie algebra elements are kept in memory to speed up the calculation.
        max_dim: Cutoff for the dimension of the Lie closure. If set, the algorithm may be halted
            before a full closure is obtained.
        verbosity: Verbosity level.
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
        return impl_sparse.lie_closure(generators, max_dim=max_dim, verbosity=verbosity,
                                       min_tasks=min_tasks, max_workers=max_workers)
    else:
        return impl_matrix.lie_closure(generators, keep_original=keep_original, max_dim=max_dim,
                                       verbosity=verbosity)
