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
    dim = elems1.shape[-1]
    vec1 = elems1.reshape(elems1.shape[:-2] + (-1,))
    vec2 = elems2.reshape(elems2.shape[:-2] + (-1,))
    ip = jnp.vecdot(vec1, vec2)
    ip += jnp.vecdot(vec1[..., dim:], vec2[..., dim:])
    return ip / dim


@partial(jax.jit, static_argnames=['skew', 'is_matrix'])
def commutator(
    op1: NDArray,
    op2: NDArray,
    skew: bool = False,
    *,
    is_matrix: tuple[bool, bool] = (False, False)
) -> NDArray[np.float64]:
    dim = op1.shape[-1]
    if not is_matrix[0]:
        op1 = to_matrix_stack(op1, skew=skew)
    if not is_matrix[1]:
        op2 = to_matrix_stack(op2, skew=skew)

    prod_real = jnp.sum(jnp.matmul(op1, op2) * jnp.array([1, -1])[:, None, None], axis=-3)
    prod_imag = jnp.sum(jnp.matmul(op1[..., ::-1, :, :], op2), axis=-3)
    rows, cols = upper_indices(dim)
    upper_real = prod_real[..., rows, cols] - prod_real[..., cols, rows]
    upper_imag = prod_imag[..., rows, cols] + prod_imag[..., cols, rows]
    extra_dims = prod_real.shape[:-2]
    upper = jnp.concatenate([upper_real, upper_imag], axis=-1).reshape(extra_dims + (dim - 1, dim))
    return jnp.concatenate([2. * jnp.diagonal(prod_imag, axis1=-2, axis2=-1)[..., None, :], upper],
                           axis=-2)


@partial(jax.jit, static_argnames=['skew'])
def to_matrix_stack(elems: NDArray[np.float64], skew: bool = False) -> Array:
    extra_dims = elems.shape[:-2]
    dim = elems.shape[-1]
    diag = elems[..., 0, :]
    upper = elems[..., 1:, :].reshape(extra_dims + (2, -1))
    matrix = jnp.zeros(extra_dims + (2, dim, dim), dtype=np.float64)

    if skew:
        lower_sign = np.array([-1., 1.])[:, None]
        idiag = 1
    else:
        lower_sign = np.array([1., -1.])[:, None]
        idiag = 0

    rows, cols = upper_indices(dim)
    matrix = matrix.at[..., rows, cols].set(upper)
    matrix = matrix.at[..., cols, rows].set(lower_sign * upper)
    didxs = np.arange(dim)
    matrix = matrix.at[..., idiag, didxs, didxs].set(diag)

    return matrix


@partial(jax.jit, static_argnames=['skew'])
def from_matrix_stack(matrix: NDArray[np.float64], skew: bool = False) -> Array:
    extra_dims = matrix.shape[:-3]
    dim = matrix.shape[-1]
    elements = jnp.zeros(extra_dims + (dim * dim,), dtype=np.float64)

    diagonals = jnp.diagonal(matrix, axis1=-2, axis2=-1)
    if skew:
        diag = diagonals[..., 1, :]
    else:
        diag = diagonals[..., 0, :]
    elements = elements.at[..., :dim].set(diag)

    rows, cols = upper_indices(dim)
    upper = matrix[..., rows, cols]
    low = dim
    high = low + len(rows)
    elements = elements.at[..., low:high].set(upper[..., 0, :])
    low = high
    high = low + len(rows)
    elements = elements.at[..., low:high].set(upper[..., 1, :])

    return elements.reshape(extra_dims + (dim, dim))


@partial(jax.jit, static_argnames=['skew'])
def to_complex_matrix(elements: NDArray[np.float64], skew: bool = False) -> Array:
    matrix = to_matrix_stack(elements, skew=skew)
    return matrix[..., 0, :, :] + 1.j * matrix[..., 1, :, :]


@partial(jax.jit, static_argnames=['skew'])
def from_complex_matrix(matrix: NDArray[np.complex128], skew: bool = False) -> Array:
    return from_matrix_stack(jnp.stack([matrix.real, matrix.imag], axis=-3), skew=skew)


def upper_indices(dim: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.array(sum(([i] * (dim - i - 1) for i in range(dim - 1)), []))
    cols = np.array(sum((list(range(i + 1, dim)) for i in range(dim - 1)), []))
    return rows, cols
