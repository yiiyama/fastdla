# pylint: disable=unused-argument
"""JAX implementation of the Lie closure generator."""
from collections.abc import Callable, Sequence
from functools import partial
import logging
from typing import Optional
import time
import numpy as np
import jax
from jax import Array
import jax.numpy as jnp
from fastdla.linalg.matrix_ops_jax import (
    innerprod as minnerprod,
    norm as mnorm,
    commutator
)
from fastdla.linalg.hermitian_ops_jax import (
    innerprod as hinnerprod,
    norm as hnorm,
    to_matrix,
    from_matrix,
    upper_indices
)
from fastdla.algorithms.gram_schmidt import gram_schmidt

LOG = logging.getLogger(__name__)
BASIS_ALLOC_UNIT = 1024

_gs_update_generic = jax.jit(
    partial(gram_schmidt, innerprod_op=minnerprod, norm_op=mnorm, npmod=jnp)
)
_gs_update_skew = jax.jit(
    partial(gram_schmidt, innerprod_op=hinnerprod, norm_op=hnorm, npmod=jnp)
)
_vcommutator_generic = jax.jit(jax.vmap(commutator, in_axes=[0, None]))


@jax.jit
def _vcommutator_skew(lhs: Array, elems: Array) -> Array:
    dim = lhs.shape[-1]
    rhs = to_matrix(elems, skew=True)
    prod = lhs @ rhs
    result = jnp.zeros(lhs.shape[:1] + elems.shape, dtype=elems.dtype)
    result = result.at[:, :dim].set(2. * jnp.diagonal(prod, axis1=1, axis2=2).imag)
    rows, cols = upper_indices(dim)
    upper = prod[:, rows, cols]
    lowert = prod[:, cols, rows]
    low = dim
    high = low + len(rows)
    result = result.at[:, low:high].set(upper.real - lowert.real)
    low = high
    high = low + len(rows)
    result = result.at[:, low:high].set(upper.imag + lowert.imag)
    return result


def _lie_basis(
    ops: Sequence[Array],
    gs_update: Callable
) -> np.ndarray:
    """Identify a basis for the linear space spanned by ops.

    Args:
        ops: Lie algebra elements whose span to calculate the basis for.

    Returns:
        A list of linearly independent ops.
    """
    ops = jnp.asarray(ops)
    if ops.shape[0] == 0:
        raise ValueError('Cannot determine the basis of null space')

    # Initialize a list of normalized generators
    basis = jnp.zeros_like(ops)
    basis, basis_size = gs_update(ops, basis=basis, basis_size=0)
    return basis[:basis_size]


def lie_basis(
    ops: Sequence[Array],
    *,
    skew_hermitian: bool = False
) -> np.ndarray:
    if skew_hermitian:
        ops = from_matrix(ops, skew=True)
        gs_update = _gs_update_skew
    else:
        gs_update = _gs_update_generic

    basis = _lie_basis(ops, gs_update)

    if skew_hermitian:
        basis = to_matrix(basis, skew=True)
    return np.array(basis)


def _init_basis(
    generators: Sequence[Array],
    gs_update: Callable
):
    generators = jnp.asarray(generators)
    if (num_gen := generators.shape[0]) == 0:
        raise ValueError('Empty set of generators')

    # Initialize the basis
    max_size = ((num_gen - 1) // BASIS_ALLOC_UNIT + 1) * BASIS_ALLOC_UNIT
    basis = jnp.zeros((max_size,) + generators.shape[1:], dtype=generators.dtype)
    basis, basis_size = gs_update(generators, basis=basis, basis_size=0)
    generators = basis[:basis_size]
    LOG.info('Number of independent generators: %d', basis_size)
    return generators, basis


def _get_loop_body(
    print_every: int | None,
    gs_update: Callable,
    vcommutator: Callable
):
    """Make the compiled loop body function using the given algorithm."""
    log_level = LOG.getEffectiveLevel()
    if print_every is None:
        if log_level <= logging.DEBUG:
            print_every = 1
        elif log_level <= logging.INFO:  # 20
            print_every = log_level * 100
        else:
            print_every = -1

    def callback(idx_op, basis_size):
        LOG.log(log_level, 'Calculating commutators with op %d/%d', idx_op, basis_size)

    @jax.jit
    def loop_body(val: tuple) -> tuple:
        """Compute the commutator and update the basis with orthogonal components."""
        idx_op, generators, basis, basis_size = val

        if print_every > 0:
            jax.lax.cond(
                jnp.equal(idx_op % print_every, 0),
                lambda: jax.debug.callback(callback, idx_op, basis_size),
                lambda: None
            )

        comms = vcommutator(generators, basis[idx_op])
        basis, basis_size = gs_update(comms, basis=basis, basis_size=basis_size)
        # If comms overshot the size of the basis, extend the basis and repeat from the same op
        new_idx = jax.lax.select(jnp.equal(basis_size, basis.shape[0]), idx_op, idx_op + 1)
        return new_idx, generators, basis, basis_size

    return loop_body


def _compute_closure(
    generators: Array,
    basis: Array,
    max_dim: int,
    loop_body: Callable
):
    # Main loop: generate nested commutators
    idx_op = 0
    basis_size = generators.shape[0]
    while True:  # Outer loop to handle memory reallocation
        main_loop_start = time.time()
        # Main (inner) loop: iteratively compute the next commutator and update the basis based on
        # the current one
        idx_op, generators, basis, new_size = jax.lax.while_loop(
            lambda val: jnp.logical_not(
                (val[0] == val[3])  # idx_op == new_size -> commutator exhausted
                | (val[3] == max_dim)  # new_size == max_dim -> reached max dim
                | (val[3] == val[2].shape[0])  # new_size == array size -> need reallocation
            ),
            loop_body,
            (idx_op, generators, basis, basis_size)
        )

        LOG.info('Found %d new ops in %.2fs. New size %d',
                 new_size - basis_size, time.time() - main_loop_start, new_size)

        basis_size = new_size
        if idx_op == basis_size or basis_size == max_dim:
            # Computed all commutators
            break

        # Need to resize basis and xmat
        LOG.debug('Resizing basis array to %d', basis.shape[0] + BASIS_ALLOC_UNIT)
        basis = jnp.pad(basis, ((0, BASIS_ALLOC_UNIT), (0, 0), (0, 0)))

    return basis[:basis_size]


def lie_closure(
    generators: Sequence[Array],
    *,
    max_dim: Optional[int] = None,
    print_every: Optional[int] = None,
    skew_hermitian: bool = False
) -> Array:
    """Compute the Lie closure of given generators.

    Args:
        generators: Lie algebra elements to compute the closure from.
        max_dim: Cutoff for the dimension of the Lie closure. If set, the algorithm may be halted
            before a full closure is obtained.

    Returns:
        A basis of the Lie algebra spanned by the nested commutators of the generators.
    """
    if skew_hermitian:
        generators = from_matrix(generators, skew=True)
        gs_update = _gs_update_skew
        vcommutator = _vcommutator_skew
    else:
        gs_update = _gs_update_generic
        vcommutator = _vcommutator_generic

    generators, basis = _init_basis(generators, gs_update)
    if skew_hermitian:
        generators = to_matrix(generators, skew=True)

    if generators.shape[0] <= 1:
        return generators

    # Make the main loop function
    loop_body = _get_loop_body(print_every, gs_update, vcommutator)
    # Run the loop
    return _compute_closure(generators, basis, max_dim, loop_body)
