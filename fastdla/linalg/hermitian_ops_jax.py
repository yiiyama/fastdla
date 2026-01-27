"""Operations for skew-Hermitian matrices.

An nxn (skew-)Hermitian matrix is represented by its diagonals (diag(M) or -i diag(M); n floats)
and the upper triangle excluding the diagonal (n(n-1)/2 complexes = n(n-1) floats) arranged into an
(n, n) array of floats.
"""
from functools import partial
import numpy as np
from numpy.typing import NDArray
import jax
from jax import Array
import jax.numpy as jnp


@jax.jit
def innerprod(elems1: NDArray[np.float64], elems2: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Inner product between two (stacked) (skew-)Hermitian matrices.

    For (skew-)Hermitian matrices A and B,

    .. math::

        \langle A, B \rangle & = \mathrm{tr} (A^{\dagger} B) \\
                             & = \sum_{ij} A^{*}_{ij} B_{ij} \\
                             & = (-1)^s \sum_{i} A_{ii} B_{ii}
                                 + \sum_{i \neq j} A^{*}_{ij} B_{ij} \\
                             & = (-1)^s \sum_{i} A_{ii} B_{ii}
                                 + 2 \mathrm{Re} \sum_{i < j} A^{*}_{ij} B_{ij} \\
                             & = (-1)^ s \sum_{i} A_{ii} B_{ii}
                                 + 2 \sum_{i < j} ( \mathrm{Re} A_{ij} \mathrm{Re} B_{ij}
                                                   + \mathrm{Im} A_{ij} \mathrm{Im} B_{ij} )

    and therefore under the representation op=[(-i)diag]+[upper real]+[upper imag], innerprod is
    given by op1 @ op2 + op1[n:] @ op2[n:].
    """
    ip = jnp.tensordot(elems1, elems2, [[-2, -1], [-2, -1]])
    ip += jnp.tensordot(elems1[..., 1:, :], elems2[..., 1:, :], [[-2, -1], [-2, -1]])
    return ip / elems1.shape[-1]


@jax.jit
def norm(elems: NDArray[np.float64]) -> NDArray[np.float64]:
    shape = elems.shape
    dim = shape[-1]
    vec = elems.reshape(shape[:-2] + (-1,))
    ip = jnp.vecdot(vec, vec)
    ip += jnp.vecdot(vec[..., dim:], vec[..., dim:])
    return jnp.sqrt(ip / dim)


@partial(jax.jit, static_argnames=['skew'])
def to_matrix(elems: NDArray[np.float64], skew: bool = False) -> Array:
    shape = elems.shape
    dim = shape[-1]
    diag = elems[..., 0, :]
    upper = elems[..., 1:, :].reshape(shape[:-2] + (-1,))
    matrix = jnp.zeros(shape, dtype=np.complex128)

    didxs = np.arange(dim)
    phase = 1.j if skew else 1.
    matrix = matrix.at[..., didxs, didxs].set(phase * diag)

    rows, cols = upper_indices(dim)
    real = upper[..., :len(rows)]
    imag = upper[..., len(rows):]
    matrix = matrix.at[..., rows, cols].set(real + 1.j * imag)
    sign = -1. if skew else 1.
    matrix = matrix.at[..., cols, rows].set(sign * (real - 1.j * imag))

    return matrix


@partial(jax.jit, static_argnames=['skew'])
def from_matrix(matrix: NDArray[np.complex128], skew: bool = False) -> Array:
    shape = matrix.shape
    dim = shape[-1]
    elements = jnp.zeros(shape[:-2] + (dim * dim,), dtype=np.float64)

    diagonals = jnp.diagonal(matrix, axis1=-2, axis2=-1)
    if skew:
        diag = diagonals.imag
    else:
        diag = diagonals.real
    elements = elements.at[..., :dim].set(diag)

    rows, cols = upper_indices(dim)
    upper = matrix[..., rows, cols]
    low = dim
    high = low + len(rows)
    elements = elements.at[..., low:high].set(upper.real)
    low = high
    high = low + len(rows)
    elements = elements.at[..., low:high].set(upper.imag)

    return elements.reshape(shape)


def upper_indices(dim: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.array(sum(([i] * (dim - i - 1) for i in range(dim - 1)), []))
    cols = np.array(sum((list(range(i + 1, dim)) for i in range(dim - 1)), []))
    return rows, cols
