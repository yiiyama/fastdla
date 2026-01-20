"""Operations for skew-Hermitian matrices.

An nxn skew-Hermitian matrix is represented by the imaginary part of its diagonals (-i diag(M);
n floats) and the upper triangle excluding the diagonal (n(n-1)/2 complexes).
"""
from typing import Optional
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
def normalize(diag: NDArray[np.float64], upper: NDArray[np.complex128]) -> tuple[Array, Array]:
    """Normalize a Hermitian matrix."""
    norm = jnp.sqrt(innerprod(diag, upper, diag, upper))
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


@jax.jit
def compose_hermitian(diag: NDArray[np.float64], upper: NDArray[np.complex128]) -> Array:
    if (squeeze := len(diag.shape) == 1):
        diag = diag[None, ...]
        upper = upper[None, ...]

    size = diag.shape[0]
    dim = diag.shape[-1]
    matrix = jnp.zeros((size, dim, dim), dtype=upper.dtype)
    # pylint: disable-next=unbalanced-tuple-unpacking
    midxs, rows, cols = upper_indices(dim, size)
    matrix = matrix.at[midxs, rows, cols].set(upper)
    matrix += matrix.conjugate().transpose((0, 2, 1))
    didxs = np.tile(np.arange(dim)[None, ...], (size, 1))
    midxs = np.repeat(np.arange(size)[:, None], dim, axis=1)
    matrix = matrix.at[midxs, didxs, didxs].set(diag)
    if squeeze:
        return matrix[0]
    return matrix


def upper_indices(dim: int, size: Optional[int] = None) -> tuple[np.ndarray, ...]:
    rows = np.array(sum(([i] * (dim - i - 1) for i in range(dim - 1)), []))
    cols = np.array(sum((list(range(i + 1, dim)) for i in range(dim - 1)), []))
    if size is None:
        return rows, cols

    rows = np.tile(rows[None, ...], (size, 1))
    cols = np.tile(cols[None, ...], (size, 1))
    midxs = np.repeat(np.arange(size)[:, None], rows.shape[-1], axis=1)
    return midxs, rows, cols
