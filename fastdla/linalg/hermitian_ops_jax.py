"""Operations for skew-Hermitian matrices.

An nxn (skew-)Hermitian matrix is represented by its diagonals (diag(M) or i diag(M); n floats) and
the upper triangle excluding the diagonal (n(n-1)/2 complexes).
"""
from functools import partial
import numpy as np
from numpy.typing import NDArray
import jax
from jax import Array
import jax.numpy as jnp


@jax.jit
def innerprod(
    diag1: NDArray[np.float64],
    upper1: NDArray[np.complex128],
    diag2: NDArray[np.float64],
    upper2: NDArray[np.complex128]
) -> Array:
    """Inner product between two (stacked) Hermitian matrices."""
    ip = jnp.tensordot(diag1, diag2, [[-1], [-1]])
    ip += jnp.tensordot(upper1.conjugate(), upper2, [[-1], [-1]]).real * 2.
    return ip / diag1.shape[-1]


@jax.jit
def frobenius_norm(diag: NDArray[np.float64], upper: NDArray[np.complex128]) -> Array:
    return jnp.sqrt(innerprod(diag, upper, diag, upper))


@jax.jit
def normalize(diag: NDArray[np.float64], upper: NDArray[np.complex128]) -> tuple[Array, Array]:
    """Normalize a Hermitian matrix."""
    norm = frobenius_norm(diag, upper)
    return jax.lax.cond(
        jnp.isclose(norm, 0.),
        lambda _diag, _upper, _norm: (jnp.zeros_like(_diag), jnp.zeros_like(_upper)),
        lambda _diag, _upper, _norm: (_diag / _norm, _upper / _norm),
        diag, upper, norm
    )


@jax.jit
def orthogonalize(
    diag: NDArray[np.float64],
    upper: NDArray[np.complex128],
    basis_diag: NDArray[np.float64],
    basis_upper: NDArray[np.complex128]
) -> Array:
    """Extract the orthogonal component of a Hermitian operator with respect to an orthonormal
    basis."""
    ip = innerprod(diag, upper, basis_diag, basis_upper)
    return (diag - jnp.tensordot(ip, basis_diag, [[0], [0]]),
            upper - jnp.tensordot(ip, basis_upper, [[0], [0]]))


@partial(jax.jit, static_argnames=['skew'])
def to_matrix(
    diag: NDArray[np.float64],
    upper: NDArray[np.complex128],
    skew: bool = False
) -> Array:
    if (squeeze := len(diag.shape) == 1):
        diag = diag[None, ...]
        upper = upper[None, ...]

    size = diag.shape[0]
    dim = diag.shape[-1]
    matrix = jnp.zeros((size, dim, dim), dtype=upper.dtype)
    # pylint: disable-next=unbalanced-tuple-unpacking
    rows, cols = upper_indices(dim)
    matrix = matrix.at[:, rows, cols].set(upper)
    if skew:
        matrix -= matrix.conjugate().transpose((0, 2, 1))
    else:
        matrix += matrix.conjugate().transpose((0, 2, 1))
    didxs = np.arange(dim)
    if skew:
        matrix = matrix.at[:, didxs, didxs].set(-1.j * diag)
    else:
        matrix = matrix.at[:, didxs, didxs].set(diag)
    if squeeze:
        return matrix[0]
    return matrix


@partial(jax.jit, static_argnames=['skew'])
def from_matrix(
    matrix: NDArray[np.complex128],
    skew: bool = False
) -> Array:
    if (squeeze := len(matrix.shape) == 2):
        matrix = matrix[None, ...]

    dim = matrix.shape[-1]
    if skew:
        diag = -jnp.diagonal(matrix, axis1=1, axis2=2).imag
    else:
        diag = jnp.diagonal(matrix, axis1=1, axis2=2).real
    # pylint: disable-next=unbalanced-tuple-unpacking
    rows, cols = upper_indices(dim)
    upper = matrix[:, rows, cols]
    if squeeze:
        return diag[0], upper[0]
    return diag, upper


def upper_indices(dim: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.array(sum(([i] * (dim - i - 1) for i in range(dim - 1)), []))
    cols = np.array(sum((list(range(i + 1, dim)) for i in range(dim - 1)), []))
    return rows, cols
