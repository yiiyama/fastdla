"""Generic matrix operations in JAX."""
import jax
from jax import Array
import jax.numpy as jnp


@jax.jit
def innerprod(op1: Array, op2: Array) -> Array:
    """Inner product between two (stacked) matrices defined by Tr(A†B)/d."""
    return jnp.tensordot(op1.conjugate(), op2, [[-2, -1], [-2, -1]]) / op1.shape[-1]


@jax.jit
def norm(op: Array) -> Array:
    shape = op.shape
    vec = op.reshape(shape[:-2] + (-1,))
    return jnp.sqrt(jnp.vecdot(vec, vec) / shape[-1])


@jax.jit
def normalize(op: Array, cutoff: float = 1.e-8) -> Array:
    """Normalize a matrix."""
    op_norm = norm(op)[..., None]
    is_null = jnp.isclose(op_norm, 0., atol=cutoff)
    return (jnp.where(is_null, 0., op) / jnp.where(is_null, 1., op_norm),
            jnp.where(is_null[..., 0], 0., op_norm[..., 0]))


@jax.jit
def project(
    op: Array,
    basis: Array
) -> Array:
    """Extract the orthogonal component of an operator with respect to an orthonormal basis."""
    # What we want is for the second term is
    #   jnp.tensordot(innerprod(basis, op), basis, [[0], [0]])
    # but we instead compute the conjugate of the innerprod to reduce the number of computation
    return jnp.tensordot(innerprod(op, basis).conjugate(), basis, [[0], [0]])


@jax.jit
def commutator(op1: Array, op2: Array) -> Array:
    """Commutator [op1, op2]."""
    return op1 @ op2 - op2 @ op1
