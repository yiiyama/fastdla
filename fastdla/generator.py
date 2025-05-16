"""Generator of Lie closure."""
from collections.abc import Sequence
from typing import Any, Optional
from fastdla.sparse_pauli_vector import SparsePauliVector, SparsePauliVectorArray
import fastdla._generator_impl.sparse as impl_sparse
import fastdla._generator_impl.matrix as impl_matrix

BasisType = Sequence[Any]
XMatrixType = Any


def linear_independence(
    new_op: Any,
    basis: BasisType,
    xinv: XMatrixType
) -> bool:
    """Check if the given operator is linearly independent from all other elements in the basis.

    Let the Lie algebra elements in the basis be P0, P1, ..., Pn. The basis_matrix Π is a matrix
    formed by stacking the column vectors {Pi}:
    Π = (P0, P1, ..., Pn).
    If new_op Q is linearly dependent on {Pi}, there is a column vector of coefficients
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
        xinv: Inverse of the X matrix.

    Returns:
        True if Q is linearly independent from all elements of the basis.
    """
    if isinstance(new_op, SparsePauliVector):
        return impl_sparse.linear_independence(new_op, basis, xinv)
    else:
        return impl_matrix.linear_independence(new_op, basis, xinv)


def lie_closure(
    generators: Any,
    *,
    keep_original: bool = True,
    max_dim: Optional[int] = None,
    verbosity: int = 0,
    min_tasks: int = 0,
    max_workers: Optional[int] = None
) -> BasisType:
    """Compute the Lie closure of given generators.

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
