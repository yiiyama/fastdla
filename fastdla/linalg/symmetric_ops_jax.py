"""Operations for imaginary symmetric and real anti-symmetric matrices.

An nxn pure-imaginary symmetric matrix is represented by its diagonals (-i diag(M); n floats) and
the upper triangle excluding the diagonal (n(n-1)/2 floats) arranged into a 1-dim array. An nxn real
anti-symmetric matrix is instead represented by the upper triangle excluding the diagonal (n(n-1)/2
floats).
"""
from functools import partial
import numpy as np
from numpy.typing import NDArray
import jax
import jax.numpy as jnp


@jax.jit
def innerprod_is(
    elems1: NDArray[np.float64],
    elems2: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""Inner product between two (stacked) imaginary symmetric matrices.

    For imaginary symmetric matrices A=iP and B=iQ,

    .. math::

        \langle A, B \rangle & = \mathrm{tr} (A^{\dagger} B) \\
                             & = \mathrm{tr} (PT Q) \\
                             & = \sum_{ij} P_{ij} Q_{ij} \\
                             & = \sum_{i} P_{ii} Q_{ii}
                                 + \sum_{i \neq j} P_{ij} Q_{ij} \\
                             & = \sum_{i} P_{ii} Q_{ii}
                                 + 2 \sum_{i < j} P_{ij} Q_{ij} \\

    and therefore under the representation op=[Im(diag)]+[Im(upper)], innerprod is given by
    op1 @ op2 + op1[n:] @ op2[n:].
    """
    # d(d+1)/2 = s -> d = (-1 + sqrt(1 + 8s)) / 2
    dim = np.round(-0.5 + np.sqrt(0.25 + 2. * elems1.shape[-1])).astype(int)
    ip = jnp.vecdot(elems1, elems2)
    ip += jnp.vecdot(elems1[..., dim:], elems2[..., dim:])
    return ip / dim


@jax.jit
def innerprod_ra(
    elems1: NDArray[np.float64],
    elems2: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""Inner product between two (stacked) real antisymmetric matrices.

    For real antisymmetric matrices A and B,

    .. math::

        \langle A, B \rangle & = \mathrm{tr} (A^{\dagger} B) \\
                             & = -\mathrm{tr} (A B) \\
                             & = \sum_{ij} A_{ij} B_{ij} \\
                             & = 2 \sum_{i < j} A_{ij} B_{ij} \\

    and therefore under the representation op=[upper], innerprod is given by 2 op1 @ op2.
    """
    # d(d-1)/2 = s -> d = (1 + sqrt(1 + 8s)) / 2
    dim = np.round(0.5 + np.sqrt(0.25 + 2. * elems1.shape[-1])).astype(int)
    ip = 2. * jnp.vecdot(elems1, elems2)
    return ip / dim


@partial(jax.jit, static_argnames=['is_ops'])
def commutator_ra(
    op1: NDArray[np.float64],
    op2: NDArray[np.float64],
    is_ops: bool = False
) -> NDArray[np.float64]:
    r"""Commutator between two (anti-)symmetric matrices, result encoded as real antisymmetric.

    The sign of the result depends on whether the inputs are real antisymmetric or the imaginary
    parts of imaginary symmetric matrices.
    """
    dim = op1.shape[-1]
    prod = jnp.matmul(op1, op2)
    rows, cols = upper_indices(dim)
    if is_ops:
        return prod[..., cols, rows] - prod[..., rows, cols]
    return prod[..., rows, cols] - prod[..., cols, rows]


@jax.jit
def commutator_is(
    op1: NDArray[np.float64],
    op2: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""Commutator between symmetric and antisymmetric matrices or vice versa, result encoded as
    imaginary symmetric.
    """
    dim = op1.shape[-1]
    prod = jnp.matmul(op1, op2)
    rows, cols = upper_indices(dim)
    upper = prod[..., rows, cols] + prod[..., cols, rows]
    return jnp.concatenate([2. * jnp.diagonal(prod, axis1=-2, axis2=-1), upper], axis=-1)


@jax.jit
def to_s_matrix(elems: NDArray[np.float64]) -> NDArray[np.float64]:
    extra_dims = elems.shape[:-1]
    sharding = jax.typeof(elems).sharding
    dim = np.round(-0.5 + np.sqrt(0.25 + 2. * elems.shape[-1])).astype(int)
    diag = elems[..., :dim]
    upper = elems[..., dim:]
    matrix = jnp.zeros(extra_dims + (dim, dim), dtype=np.float64, device=sharding)
    rows, cols = upper_indices(dim)
    matrix = matrix.at[..., rows, cols].set(upper)
    matrix = matrix.at[..., cols, rows].set(upper)
    didxs = np.arange(dim)
    matrix = matrix.at[..., didxs, didxs].set(diag)
    return matrix


@jax.jit
def to_a_matrix(elems: NDArray[np.float64]) -> NDArray[np.float64]:
    extra_dims = elems.shape[:-1]
    sharding = jax.typeof(elems).sharding
    dim = np.round(0.5 + np.sqrt(0.25 + 2. * elems.shape[-1])).astype(int)
    matrix = jnp.zeros(extra_dims + (dim, dim), dtype=np.float64, device=sharding)
    rows, cols = upper_indices(dim)
    matrix = matrix.at[..., rows, cols].set(elems)
    matrix = matrix.at[..., cols, rows].set(-elems)
    return matrix


@jax.jit
def from_s_matrix(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    extra_dims = matrix.shape[:-2]
    sharding = jax.typeof(matrix).sharding
    dim = matrix.shape[-1]
    elements = jnp.zeros(extra_dims + (dim * (dim + 1) // 2,), dtype=np.float64, device=sharding)
    diag = jnp.diagonal(matrix, axis1=-2, axis2=-1)
    elements = elements.at[..., :dim].set(diag)
    rows, cols = upper_indices(dim)
    upper = matrix[..., rows, cols]
    elements = elements.at[..., dim:].set(upper)
    return elements


@jax.jit
def from_a_matrix(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    dim = matrix.shape[-1]
    rows, cols = upper_indices(dim)
    elements = matrix[..., rows, cols]
    return elements


@jax.jit
def to_complex_is_matrix(elements: NDArray[np.float64]) -> NDArray[np.complex128]:
    return 1.j * to_s_matrix(elements)


@jax.jit
def from_complex_is_matrix(matrix: NDArray[np.complex128]) -> NDArray[np.float64]:
    return from_s_matrix(matrix.imag)


@jax.jit
def to_complex_ra_matrix(elements: NDArray[np.float64]) -> NDArray[np.complex128]:
    return to_a_matrix(elements).astype(np.complex128)


@jax.jit
def from_complex_ra_matrix(matrix: NDArray[np.complex128]) -> NDArray[np.float64]:
    return from_a_matrix(matrix.real)


def upper_indices(dim: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.array(sum(([i] * (dim - i - 1) for i in range(dim - 1)), []))
    cols = np.array(sum((list(range(i + 1, dim)) for i in range(dim - 1)), []))
    return rows, cols
