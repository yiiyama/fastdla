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
def commutator(op1: Array, op2: Array) -> Array:
    """Commutator [op1, op2]."""
    return op1 @ op2 - op2 @ op1
