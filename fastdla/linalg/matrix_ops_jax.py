"""Generic matrix operations in JAX."""
from typing import Optional
import numpy as np
import jax
from jax import Array
import jax.numpy as jnp


@jax.jit
def innerprod(op1: Array, op2: Array) -> Array:
    """Inner product between two (stacked) matrices defined by Tr(A†B)/d."""
    return jnp.tensordot(op1.conjugate(), op2, [[-2, -1], [-2, -1]]) / op1.shape[-1]


@jax.jit
def innerprodh(diag1: Array, upper1: Array, diag2: Array, upper2: Array) -> Array:
    """Inner product between two (stacked) Hermitian matrices."""
    ip = jnp.tensordot(upper1.conjugate(), upper2, [[-1], [-1]]).real * 2.
    ip += jnp.tensordot(diag1.conjugate(), diag2, [[-1], [-1]])
    return ip / diag1.shape[-1]


@jax.jit
def normalize(op: Array) -> Array:
    """Normalize a matrix."""
    norm = jnp.sqrt(innerprod(op, op))
    return jax.lax.cond(
        jnp.isclose(norm, 0.),
        lambda _op, _norm: jnp.zeros_like(_op),
        lambda _op, _norm: _op / _norm,
        op, norm
    )


@jax.jit
def normalizeh(diag: Array, upper: Array) -> tuple[Array, Array]:
    """Normalize a Hermitian matrix."""
    norm = jnp.sqrt(innerprodh(diag, upper, diag, upper))
    return jax.lax.cond(
        jnp.isclose(norm, 0.),
        lambda _diag, _upper, _norm: (jnp.zeros_like(_diag), jnp.zeros_like(_upper)),
        lambda _diag, _upper, _norm: (_diag / _norm, _upper / _norm),
        diag, upper, norm
    )


@jax.jit
def orthogonalize(
    op: Array,
    basis: Array
) -> Array:
    """Extract the orthogonal component of an operator with respect to an orthonormal basis."""
    # What we want is for the second term is
    #   jnp.tensordot(innerprod(basis, new_op), basis, [[0], [0]])
    # but we instead compute the conjugate of the innerprod to reduce the number of computation
    return op - jnp.tensordot(innerprod(op, basis).conjugate(), basis, [[0], [0]])


@jax.jit
def orthogonalizeh(
    diag: Array,
    upper: Array,
    basis_diag: Array,
    basis_upper: Array
) -> Array:
    """Extract the orthogonal component of a Hermitian operator with respect to an orthonormal
    basis."""
    ip = innerprodh(diag, upper, basis_diag, basis_upper).conjugate()
    return (diag - jnp.tensordot(ip, basis_diag, [[0], [0]]),
            upper - jnp.tensordot(ip, basis_upper, [[0], [0]]))


@jax.jit
def commutator(op1: Array, op2: Array) -> Array:
    """Commutator [op1, op2]."""
    return op1 @ op2 - op2 @ op1


@jax.jit
def commutatorh(op1: Array, diag2: Array, upper2: Array) -> tuple[Array, Array]:
    """Commutator between two Hermitian matrices multiplied by -i."""
    op2 = compose_hermitian(diag2, upper2)
    prod = op1 @ op2
    rows, cols = upper_indices(op1.shape[-1])
    diag = 2. * jnp.diagonal(prod).imag
    upper = -1.j * (prod[rows, cols] - prod.T[rows, cols].conjugate())
    return diag, upper


@jax.jit
def compose_hermitian(diag: Array, upper: Array) -> Array:
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
