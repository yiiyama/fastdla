"""General linear algebra routine to find an eigenspace of an operator."""
from collections.abc import Callable
from numbers import Number
from typing import Optional
import numpy as np

LinearOpFunction = Callable[[np.ndarray], np.ndarray]


def get_eigenspace(
    op: np.ndarray | LinearOpFunction | tuple[np.ndarray, Number] | tuple[LinearOpFunction, Number],
    basis: Optional[np.ndarray] = None,
    dim: Optional[int] = None,
    npmod=np
) -> np.ndarray:
    r"""Extract eigenvectors of an operator within a space spanned by a set of vectors.

    Let :math:`B` be a matrix consisting of :math:`S` linearly independent vectors of length
    :math:`N` (implying :math:`S \leq N`). A vector in the space spanned by columns of :math:`B` is
    given by :math:`Bx`, where :math:`x` is a column of S coefficients. When :math:`Bx` is an
    eigenvector of operator :math:`O` with eigenvalue :math:`\lambda`,

    .. math::
       :label: x_eigen

        O Bx = \lambda Bx \\
        \therefore (O - \lambda I)Bx = 0.

    Let the singular value decomposition of :math:`(O - \lambda I)B` be

    .. math::

        (O - \lambda I)B = U \Sigma V^{\dagger} = U \sum_{j=0}^{S-1} \sigma_j v_j^{\dagger},

    where :math:`\{v_j\}_j` are orthonormal column vectors of length :math:`S`. For Equation
    :eq:`x_eigen` to hold, we need

    .. math::

        x \in \mathrm{span}(\{v_j | \sigma_j = 0\}).

    The eigen-subspace of the original space corresponding to the eigenvalue :math:`\lambda` of
    operator :math:`O` is then :math:`\mathrm{span}(\{B v_j | \sigma_j = 0\})`.

    Note that in the above we do not require an explicit form of :math:`O` but only need the result
    of applying it on the columns of :math:`B`.

    Args:
        op: Either an operator corresponding to :math:`O - \lambda I` or a tuple
            :math:`(O, \lambda)`. The operator can be a matrix or a callable that applies an
            :math:`N \times N` linear operator to an :math:`N \times S` matrix.
        eigenvalue: Eigenvalue of the operator to find the eigenspace for.
        basis: An array of :math:`S` linearly independent column vectors of length :math:`N`. If not
            given, the identity matrix is assumed.
        dim: The dimension :math:`N` of the linear space (necessary only when `op` is a callable and
            `basis` is not given).

    Returns:
        An array of shape :math:`(N, S')`, where :math:`S'` is the dimension of the eigen-subspace.
    """
    if isinstance(op, tuple):
        op, eigenvalue = op
    else:
        eigenvalue = None

    if callable(op):
        if basis is None:
            if dim is None:
                raise ValueError('Need dimension specification')
            singular_mat = op(np.eye(dim, dtype=np.complex128))
        else:
            singular_mat = op(basis)
    else:
        if basis is None:
            singular_mat = np.array(op)
        else:
            singular_mat = op @ basis

    dim = singular_mat.shape[0]

    if eigenvalue is not None:
        if basis is None:
            singular_mat[np.arange(dim), np.arange(dim)] -= eigenvalue
        else:
            singular_mat -= eigenvalue * basis

    _, svals, vhmat = npmod.linalg.svd(singular_mat, full_matrices=False)
    indices = npmod.nonzero(npmod.isclose(svals, 0.))[0]
    v_columns = vhmat[indices].conjugate().T

    if basis is None:
        return v_columns

    return basis @ v_columns
