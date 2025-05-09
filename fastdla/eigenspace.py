"""General linear algebra routine to find an eigenspace of an operator."""
from collections.abc import Callable
from numbers import Number
import numpy as np


def get_eigenspace(
    op: np.ndarray | Callable[[np.ndarray], np.ndarray],
    eigenvalue: Number,
    subspace_basis: np.ndarray,
    npmod=np
) -> np.ndarray:
    """Extract eigenvectors of an operator within a subspace spanned by a set of vectors.

    Algorithm:
    Let B represent the given subspace basis (shape (N, S)). A vector in the space spanned by
    columns of B is given by Bx, where x is a column vector with S entries. When Bx is an
    eigenvector of operator O with eigenvalue λ,
        O Bx = λ Bx ⇒ (O - λI)Bx = 0.
    If the SVD of (O - λI)B is UΣV†, x is equal to the conjugate of a row of V† corresponding to a
    zero singular value. As there can be multiple such xs, we organize them as column vectors and
    return B [x0, x1, ...], which is a basis of the space within the initial subspace that is
    spanned by the eigenvectors of O.

    Args:
        op: A matrix or a callable that applies the N-square operator to an array of shape (N, S).
        eigenvalue: Eigenvalue of the operator to find the eigenspace for.
        subspace_basis: A set of mutually orthogonal S column vectors of length N.

    Returns:
        An array of shape (N, S'), where S' is the number of eigenvectors of op found in the
        subspace.
    """
    if callable(op):
        transformed = op(subspace_basis)
    else:
        transformed = op @ subspace_basis

    _, svals, vhmat = npmod.linalg.svd(transformed - eigenvalue * subspace_basis,
                                       full_matrices=False)
    indices = npmod.nonzero(npmod.isclose(svals, 0.))[0]
    combinations = vhmat[indices].conjugate().T

    return subspace_basis @ combinations
