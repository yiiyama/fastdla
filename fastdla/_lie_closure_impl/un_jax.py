# pylint: disable=unused-argument
"""Implementation of the Lie closure generator optimized for skew-Hermitian matrices using JAX."""
from collections.abc import Sequence
from functools import partial
import logging
from typing import Optional
import numpy as np
import jax
from jax import Array
import jax.numpy as jnp
from fastdla.linalg.hermitian_ops_jax import innerprod, norm, to_matrix, from_matrix, upper_indices
from fastdla.algorithms.gram_schmidt import gram_schmidt
from fastdla._lie_closure_impl.matrix_jax import _lie_basis, _init_basis, _compute_closure

LOG = logging.getLogger(__name__)
BASIS_ALLOC_UNIT = 1024

_gs_update = jax.jit(
    partial(gram_schmidt, innerprod_op=innerprod, norm_op=norm, npmod=jnp)
)


@jax.jit
@partial(jax.vmap, in_axes=[0, None])
@jax.jit
def _vcommutator(lhs: Array, elems: Array) -> Array:
    dim = lhs.shape[-1]
    rhs = to_matrix(elems, skew=True)
    prod = lhs @ rhs
    result = jnp.zeros_like(elems)
    result = result.at[:dim].set(2. * jnp.diagonal(prod).imag)
    rows, cols = upper_indices(dim)
    low = dim
    high = low + len(rows)
    result = result.at[low:high].set(prod[rows, cols].real - prod[cols, rows].real)
    low = high
    high = low + len(rows)
    result = result.at[low:high].set(prod[rows, cols].imag + prod[cols, rows].imag)
    return result


def lie_basis(
    ops: Sequence[np.ndarray]
) -> np.ndarray:
    """Identify a basis for the linear space spanned by ops.

    Args:
        ops: Lie algebra elements whose span to calculate the basis for.

    Returns:
        A list of linearly independent ops.
    """
    elements = from_matrix(ops, skew=True)
    basis = _lie_basis(elements, _gs_update)
    return np.array(to_matrix(basis, skew=True))


def lie_closure(
    generators: Sequence[np.ndarray],
    *,
    max_dim: Optional[int] = None,
    print_every: Optional[int] = None
) -> np.ndarray | tuple[np.ndarray, list]:
    """Compute the Lie closure of given generators.

    Args:
        generators: Lie algebra elements to compute the closure from.
        max_dim: Cutoff for the dimension of the Lie closure. If set, the algorithm may be halted
            before a full closure is obtained.

    Returns:
        A basis of the Lie algebra spanned by the nested commutators of the generators.
    """
    gen_elems = from_matrix(generators, skew=True)
    gen_elems, basis = _init_basis(gen_elems, _gs_update)
    generators = to_matrix(gen_elems)
    if generators.shape[0] <= 1:
        return generators

    return _compute_closure(generators, basis, _gs_update, _vcommutator, max_dim, print_every)
