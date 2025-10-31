"""Generic matrix operations in numpy/numba."""
import numpy as np
from numba import njit


@njit(nogil=True, inline='always')
def complex_isclose(lhs, rhs, rtol=1.e-5, atol=1.e-8, equal_nan=False):
    return (np.isclose(lhs.real, rhs.real, rtol=rtol, atol=atol, equal_nan=equal_nan)
            and np.isclose(lhs.imag, rhs.imag, rtol=rtol, atol=atol, equal_nan=equal_nan))


@njit(nogil=True, inline='always')
def abs_square(value):
    return np.square(value.real) + np.square(value.imag)


@njit(nogil=True, inline='always')
def _check_finite(coeff, atol_real, atol_imag, iout, normsq):
    """Check for rounding errors in cancellations.

    *WARNING* This procedure can eliminate legitimately small result in cases like a sum of two
    large and cancelling values and a small value.
    """
    atol = max(atol_real, atol_imag) * 1.e-5
    if complex_isclose(coeff, 0., atol=atol):
        return iout, normsq
    return iout + 1, normsq + abs_square(coeff)


@njit(nogil=True, inline='always')
def innerprod(op1: np.ndarray, op2: np.ndarray) -> complex:
    """Inner product between two matrices defined by Tr(A†B)/d. A can be a stack of matrices."""
    result = np.zeros(op1.shape[:-2], dtype=op1.dtype)
    op1_star = op1.conjugate()
    for ix in range(op2.shape[-1]):
        result += np.sum(op1_star[..., ix] * op2[:, ix], axis=-1)
    result /= op2.shape[-1]
    return result


@njit
def normalize(op: np.ndarray) -> np.ndarray:
    """Normalize a matrix."""
    norm = np.sqrt(innerprod(op, op).real)
    if np.isclose(norm, 0.):
        return np.zeros_like(op)
    return op / norm


@njit(nogil=True, inline='always')
def orthogonalize(
    new_op: np.ndarray,
    basis: np.ndarray
) -> np.ndarray:
    """Extract the orthogonal component of an operator with respect to an orthonormal basis."""
    ips = innerprod(basis, new_op)
    result = new_op - np.sum(ips[:, None, None] * basis, axis=0)
    return result


@njit(nogil=True, inline='always')
def commutator(op1: np.ndarray, op2: np.ndarray) -> np.ndarray:
    """Commutator [op1, op2]."""
    return op1 @ op2 - op2 @ op1
