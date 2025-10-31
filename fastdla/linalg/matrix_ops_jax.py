"""Generic matrix operations in JAX."""
import jax
from jax import Array
import jax.numpy as jnp


@jax.jit
def innerprod(op1: Array, op2: Array) -> complex:
    """Inner product between two (stacked) matrices defined by Tr(A†B)/d."""
    return jnp.tensordot(op1.conjugate(), op2, [[-2, -1], [-2, -1]]) / op1.shape[-1]


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
def orthogonalize(
    new_op: Array,
    basis: Array
) -> Array:
    """Extract the orthogonal component of an operator with respect to an orthonormal basis."""
    return new_op - jnp.tensordot(innerprod(basis, new_op), basis, [[0], [0]])


@jax.jit
def commutator(op1: Array, op2: Array) -> Array:
    """Commutator [op1, op2]."""
    return op1 @ op2 - op2 @ op1
