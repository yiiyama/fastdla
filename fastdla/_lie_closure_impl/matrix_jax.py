# pylint: disable=unused-argument
"""JAX implementation of the Lie closure generator."""
from collections.abc import Sequence
from functools import partial
import logging
from typing import Optional
import time
import numpy as np
import jax
from jax import Array
import jax.numpy as jnp
from fastdla.linalg.matrix_ops_jax import commutator, normalize, project
from fastdla.algorithms.gram_schmidt import gram_schmidt

LOG = logging.getLogger(__name__)
BASIS_ALLOC_UNIT = 1024


def get_loop_body(print_every: Optional[int] = None):
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

    vcommutator = jax.jit(jax.vmap(commutator, in_axes=[0, None]))
    gs_update = jax.jit(
        partial(gram_schmidt, project_op=project, normalize_op=normalize, npmod=jnp)
    )

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


def lie_basis(
    ops: Sequence[Array]
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
    basis, basis_size = gram_schmidt(ops, basis=basis, basis_size=0, project_op=project,
                                     normalize_op=normalize, npmod=jnp)
    return np.array(basis[:basis_size])


def lie_closure(
    generators: Sequence[Array],
    *,
    max_dim: Optional[int] = None,
    print_every: Optional[int] = None
) -> Array:
    """Compute the Lie closure of given generators.

    Args:
        generators: Lie algebra elements to compute the closure from.
        max_dim: Cutoff for the dimension of the Lie closure. If set, the algorithm may be halted
            before a full closure is obtained.

    Returns:
        A basis of the Lie algebra spanned by the nested commutators of the generators.
    """
    generators = jnp.asarray(generators)
    if (num_gen := generators.shape[0]) == 0:
        raise ValueError('Empty set of generators')

    # Initialize the basis
    max_size = ((num_gen - 1) // BASIS_ALLOC_UNIT + 1) * BASIS_ALLOC_UNIT
    basis = jnp.zeros((max_size,) + generators.shape[1:], dtype=generators.dtype)
    basis, basis_size = gram_schmidt(generators, basis=basis, basis_size=0, project_op=project,
                                     normalize_op=normalize, npmod=jnp)
    generators = basis[:basis_size]
    num_gen = basis_size
    LOG.info('Number of independent generators: %d', num_gen)
    if num_gen <= 1:
        return generators

    # Get the main loop function for the algorithm
    loop_body = get_loop_body(print_every=print_every)

    # Main loop: generate nested commutators
    idx_op = 0
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
        LOG.debug('Resizing basis array to %d', max_size + BASIS_ALLOC_UNIT)

        max_size += BASIS_ALLOC_UNIT
        basis = jnp.pad(basis, ((0, BASIS_ALLOC_UNIT), (0, 0), (0, 0)))

    return basis[:basis_size]
