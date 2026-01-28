"""Generic matrix operations in JAX."""
import jax
from jax import Array
import jax.numpy as jnp


@jax.jit
def innerprod(op1: Array, op2: Array) -> Array:
    """Inner product between two (stacked) matrices defined by Tr(A†B)/d."""
    vec1 = op1.reshape(op1.shape[:-2] + (-1,))
    vec2 = op2.reshape(op2.shape[:-2] + (-1,))
    return jnp.vecdot(vec1, vec2) / op1.shape[-1]


@jax.jit
def commutator(op1: Array, op2: Array) -> Array:
    """Commutator [op1, op2]."""
    return jnp.matmul(op1, op2) - jnp.matmul(op2, op1)
