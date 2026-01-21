# pylint: disable=unused-argument
"""Implementation of the Lie closure generator optimized for (anti-)Hermitian matrices using JAX."""
from collections.abc import Sequence
from functools import partial
import logging
from typing import Optional
import time
import numpy as np
import jax
from jax import Array
import jax.numpy as jnp
from fastdla.linalg.hermitian_ops_jax import (innerprod, normalize, orthogonalize, to_matrix,
                                              from_matrix, upper_indices)
from fastdla._lie_closure_impl.algorithms import Algorithms

LOG = logging.getLogger(__name__)
BASIS_ALLOC_UNIT = 1024


@jax.jit
def _has_orthcomp(
    diag: Array,
    upper: Array,
    basis_diag: Array,
    basis_upper: Array
) -> tuple[bool, Array, Array]:
    orth_diag, orth_upper = orthogonalize(diag, upper, basis_diag, basis_upper)
    orth_diag, orth_upper = normalize(orth_diag, orth_upper)
    orth_diag, orth_upper = orthogonalize(orth_diag, orth_upper, basis_diag, basis_upper)
    norm = jnp.sqrt(innerprod(orth_diag, orth_upper, orth_diag, orth_upper))
    return jnp.isclose(norm, 1., rtol=1.e-5), orth_diag / norm, orth_upper / norm


@partial(jax.jit, static_argnums=[0])
def _zeros_with_entries(size: int, init_diag: Array, init_upper: Array) -> tuple[Array, Array]:
    diag = jnp.zeros((size, init_diag.shape[1]), dtype=init_diag.dtype)
    upper = jnp.zeros((size, init_upper.shape[1]), dtype=init_upper.dtype)
    return (diag.at[:init_diag.shape[0]].set(init_diag),
            upper.at[:init_upper.shape[0]].set(init_upper))


@jax.jit
def _update_gs_direct(
    diag: Array,
    upper: Array,
    pos: int,
    basis_diag: Array,
    basis_upper: Array
) -> tuple[bool, Array, Array]:
    """Update the basis if op has an orthogonal component."""
    def _update(_orth_diag, _orth_upper, _basis_diag, _basis_upper, _pos):
        return (_basis_diag.at[_pos].set(_orth_diag), _basis_upper.at[_pos].set(_orth_upper))

    has_orthcomp, orth_diag, orth_upper = _has_orthcomp(diag, upper, basis_diag, basis_upper)
    basis_diag, basis_upper = jax.lax.cond(
        has_orthcomp,
        _update,
        lambda a, b, _basis_diag, _basis_upper, c: (_basis_diag, _basis_upper),
        orth_diag, orth_upper, basis_diag, basis_upper, pos
    )
    return has_orthcomp, basis_diag, basis_upper


@jax.jit
def _update_basis(
    diag: Array,
    upper: Array,
    basis_size: int,
    basis_diag: Array,
    basis_upper: Array
) -> tuple[int, Array, Array]:
    def update_with_norm(_diag, _upper, _basis_size, _basis_diag, _basis_upper):
        _diag, _upper = normalize(_diag, _upper)
        up, bd, bu = _update_gs_direct(_diag, _upper, _basis_size, _basis_diag, _basis_upper)
        return jax.lax.select(up, _basis_size + 1, _basis_size), bd, bu

    # If non-null, call the updater function
    return jax.lax.cond(
        jnp.logical_and(jnp.allclose(basis_diag, 0.), jnp.allclose(basis_upper, 0.)),
        lambda a, b, bs, bd, bu: (bs, bd, bu),
        update_with_norm,
        diag, upper, basis_size, basis_diag, basis_upper
    )


def get_loop_body(matrix_dim: int, print_every: Optional[int] = None):
    """Make the compiled loop body function using the given algorithm."""
    log_level = LOG.getEffectiveLevel()
    if print_every is None:
        if log_level <= logging.DEBUG:
            print_every = 1
        elif log_level <= logging.INFO:  # 20
            print_every = log_level * 100
        else:
            print_every = -1

    def callback(idx_gen, idx_op, basis_size, generators):
        icomm = idx_op * generators.shape[0] + idx_gen
        LOG.log(log_level,
                'Basis size %d; %dth/%d commutator [g[%d], b[%d]]',
                basis_size, icomm, basis_size * generators.shape[0], idx_gen, idx_op)

    rows, cols = upper_indices(matrix_dim)
    to_skew_matrix = partial(to_matrix, skew=True)

    @jax.jit
    def comm_and_update(generator, op, basis_size, basis_diag, basis_upper):
        prod = generator @ op
        diag = -2. * jnp.diagonal(prod).imag
        upper = prod[rows, cols] - prod.T[rows, cols].conjugate()

        return _update_basis(diag, upper, basis_size, basis_diag, basis_upper)

    @jax.jit
    def loop_body(val: tuple) -> tuple:
        """Compute the commutator and update the basis with orthogonal components."""
        idx_gen, idx_op, generators, basis_size, basis_diag, basis_upper, op = val

        if print_every > 0:
            icomm = idx_op * generators.shape[0] + idx_gen
            jax.lax.cond(
                jnp.equal(icomm % print_every, 0),
                lambda: jax.debug.callback(callback, idx_gen, idx_op, basis_size, generators),
                lambda: None
            )

        basis_size, basis_diag, basis_upper = comm_and_update(generators[idx_gen], op, basis_size,
                                                              basis_diag, basis_upper)

        # Compute the next indices
        idx_gen = (idx_gen + 1) % generators.shape[0]
        next_op = jnp.equal(idx_gen, 0)
        idx_op = jax.lax.select(next_op, idx_op + 1, idx_op)
        op = jax.lax.cond(next_op, to_skew_matrix, lambda d, u: op,
                          basis_diag[idx_op], basis_upper[idx_op])

        return idx_gen, idx_op, generators, basis_size, basis_diag, basis_upper, op

    return comm_and_update, loop_body


def _lie_basis(
    ops: Sequence[np.ndarray]
) -> tuple[Array, Array]:
    ops = jnp.array(ops)

    nops = ops.shape[0]
    if nops == 0:
        raise ValueError('Cannot determine the basis of null space')

    diag, upper = from_matrix(ops, skew=True)
    first_diag, first_upper = normalize(diag[0], upper[0])

    # Initialize a list of normalized generators
    basis_diag, basis_upper = _zeros_with_entries(nops, first_diag[None, :], first_upper[None, :])

    def _update(iop, val):
        bsize, bdiag, bupper = val
        return _update_basis(diag[iop], upper[iop], bsize, bdiag, bupper)

    size, basis_diag, basis_upper = jax.lax.fori_loop(
        1, nops,
        _update,
        (1, basis_diag, basis_upper)
    )
    return basis_diag[:size], basis_upper[:size]


@partial(jax.jit, static_argnames=['max_size'])
def _resize_basis(
    basis_diag: Array,
    basis_upper: Array,
    max_size: int
):
    """Expand the arrays to a new size."""
    old_size = basis_diag.shape[0]
    new_shape = (max_size,) + basis_diag.shape[1:]
    basis_diag = jnp.zeros(new_shape, dtype=basis_diag.dtype).at[:old_size].set(basis_diag)
    new_shape = (max_size,) + basis_upper.shape[1:]
    basis_upper = jnp.zeros(new_shape, dtype=basis_upper.dtype).at[:old_size].set(basis_upper)
    return basis_diag, basis_upper


def lie_basis(
    ops: Sequence[np.ndarray],
    *,
    algorithm: Algorithms = Algorithms.GS_DIRECT,
) -> np.ndarray:
    """Identify a basis for the linear space spanned by ops.

    Args:
        ops: Lie algebra elements whose span to calculate the basis for.

    Returns:
        A list of linearly independent ops.
    """
    if algorithm != Algorithms.GS_DIRECT:
        raise NotImplementedError('Only GS_DIRECT algorithm is available')

    diag, upper = _lie_basis(ops)
    return np.array(to_matrix(diag, upper, skew=True))


def lie_closure(
    generators: Sequence[np.ndarray],
    *,
    max_dim: Optional[int] = None,
    algorithm: Algorithms = Algorithms.GS_DIRECT,
    return_aux: bool = False,
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
    if algorithm != Algorithms.GS_DIRECT:
        raise NotImplementedError('Only GS_DIRECT algorithm is available')

    gen_diag, gen_upper = _lie_basis(generators)
    generators = to_matrix(gen_diag, gen_upper, skew=True)
    LOG.info('Number of independent generators: %d', generators.shape[0])
    num_gen = generators.shape[0]
    if num_gen <= 1:
        return generators

    # Get the main loop function for the algorithm
    comm_and_update, main_loop_body = get_loop_body(generators.shape[-1], print_every=print_every)
    max_dim = max_dim or generators.shape[-1] ** 2

    # Initialize the basis
    max_size = ((num_gen - 1) // BASIS_ALLOC_UNIT + 1) * BASIS_ALLOC_UNIT
    basis_diag, basis_upper = _zeros_with_entries(max_size, gen_diag, gen_upper)

    # First compute the commutators among the generators
    def set_initial_comms(it, val):
        _idx_gens, _idx_ops, _generators, _basis_size, _basis_diag, _basis_upper = val
        idx_gen = _idx_gens[it]
        idx_op = _idx_ops[it]
        generator = _generators[idx_gen]
        op = to_matrix(_basis_diag[idx_op], _basis_upper[idx_op], skew=True)
        _basis_size, _basis_diag, _basis_upper = comm_and_update(generator, op, _basis_size,
                                                                 _basis_diag, _basis_upper)
        return (_idx_gens, _idx_ops, _generators, _basis_size, _basis_diag, _basis_upper)

    idx_gens, idx_ops = np.array(
        [[ig, io] for io in range(num_gen) for ig in range(io)]
    ).T
    basis_size, basis_diag, basis_upper = jax.lax.fori_loop(
        0, len(idx_gens),
        set_initial_comms,
        (idx_gens, idx_ops, generators, num_gen, basis_diag, basis_upper)
    )[3:]

    # This would be stupid but possible
    if basis_size >= max_dim:
        return to_matrix(basis_diag[:basis_size], basis_upper[:basis_size], skew=True)

    # Main loop: generate nested commutators
    idx_gen = 0
    idx_op = num_gen
    op = to_matrix(basis_diag[idx_op], basis_upper[idx_op], skew=True)
    while True:  # Outer loop to handle memory reallocation
        LOG.info('Current Lie algebra dimension: %d', basis_size)
        main_loop_start = time.time()
        # Main (inner) loop: iteratively compute the next commutator and update the basis based on
        # the current one
        idx_gen, idx_op, generators, new_size, basis_diag, basis_upper, op = jax.lax.while_loop(
            lambda val: jnp.logical_not(
                (val[1] == val[3])  # idx_op == new_size -> commutator exhausted
                | (val[3] == max_dim)  # new_size == max_dim -> reached max dim
                | (val[3] == val[4].shape[0])  # new_size == array size -> need reallocation
            ),
            main_loop_body,
            (idx_gen, idx_op, generators, basis_size, basis_diag, basis_upper, op)
        )

        LOG.info('Found %d new ops in %.2fs',
                 new_size - basis_size, time.time() - main_loop_start)

        basis_size = new_size
        if idx_op == basis_size or basis_size == max_dim:
            # Computed all commutators
            break

        # Need to resize basis and xmat
        LOG.debug('Resizing basis array to %d', max_size + BASIS_ALLOC_UNIT)

        max_size += BASIS_ALLOC_UNIT
        basis_diag, basis_upper = _resize_basis(basis_diag, basis_upper, max_size)

    return to_matrix(basis_diag[:basis_size], basis_upper[:basis_size], skew=True)
