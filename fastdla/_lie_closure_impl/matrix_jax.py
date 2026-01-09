# pylint: disable=unused-argument
"""Implementation of the Lie closure generator using JAX matrices."""
from collections.abc import Sequence
from functools import partial
import logging
from typing import Optional
import time
import numpy as np
import jax
from jax import Array
import jax.numpy as jnp
from fastdla.linalg.matrix_ops_jax import commutator, innerprod, normalize, orthogonalize
from fastdla._lie_closure_impl.algorithms import Algorithms

LOG = logging.getLogger(__name__)
BASIS_ALLOC_UNIT = 1024


@jax.jit
def _has_orthcomp(op: Array, basis: Array) -> tuple[bool, Array]:
    orth = orthogonalize(normalize(orthogonalize(op, basis)), basis)
    norm = jnp.sqrt(innerprod(orth, orth))
    return jnp.isclose(norm, 1., rtol=1.e-5), orth / norm


@partial(jax.jit, static_argnums=[0])
def _zeros_with_entries(size: int, init: Array) -> Array:
    array = jnp.zeros((size,) + init.shape[1:], dtype=init.dtype)
    return array.at[:init.shape[0]].set(init)


@jax.jit
def _update_gs_direct(
    op: Array,
    size: int,
    basis: Array
) -> tuple[bool, Array]:
    """Update the basis if op has an orthogonal component."""
    has_orthcomp, orth = _has_orthcomp(op, basis)
    basis = jax.lax.cond(
        has_orthcomp,
        lambda _orth, _basis, _size: _basis.at[_size].set(_orth),
        lambda a, _basis, b: _basis,
        orth, basis, size
    )
    return has_orthcomp, basis


@jax.jit
def _update_gram_schmidt(
    op: Array,
    size: int,
    basis: Array,
    orthonormal_basis: Array
) -> tuple[bool, Array, Array]:
    """Update the basis and the nested commutators list if op has an orthogonal component."""
    has_orthcomp, orthonormal_basis = _update_gs_direct(op, size, orthonormal_basis)
    basis = jax.lax.cond(
        has_orthcomp,
        lambda _op, _basis, _size: _basis.at[_size].set(_op),
        lambda a, _basis, b: _basis,
        op, basis, size
    )
    return has_orthcomp, basis, orthonormal_basis


@jax.jit
def _update_matrix_inv(
    op: Array,
    size: int,
    basis: Array,
    xmat: Array,
    xinv: Array
) -> tuple[bool, Array, Array, Array]:
    """Update the basis and the X matrix with op if it is independent."""
    def _residual(_op, _basis, _xinv, _pidag_q):
        # Residual calculation: subtract Pi*ai from Q directly
        a_proj = _xinv @ _pidag_q
        residual = _op - jnp.sum(_basis * a_proj[:, None, None], axis=0)
        return jnp.logical_not(jnp.allclose(residual, 0.)), _pidag_q

    # Compute the Π†Q vector
    pidag_q = innerprod(basis, op)
    # If pidag_q is non-null, compute the residual
    is_independent, new_xcol = jax.lax.cond(
        jnp.allclose(pidag_q, 0.),
        lambda a, b, c, _pidag_q: (True, _pidag_q),
        _residual,
        op, basis, xinv, pidag_q
    )

    def _update(_op, _new_xcol, _basis, _size, _xmat, _):
        _basis = _basis.at[_size].set(_op)
        _xmat = _xmat.at[:, _size].set(_new_xcol).at[size, :].set(_new_xcol.conjugate())
        _xmat = _xmat.at[_size, _size].set(1.)
        return _basis, _xmat, jnp.linalg.inv(_xmat)

    basis, xmat, xinv = jax.lax.cond(
        is_independent,
        _update,
        lambda a, b, _basis, _size, _xmat, _xinv: (_basis, _xmat, _xinv),
        op, new_xcol, basis, size, xmat, xinv
    )
    return is_independent, basis, xmat, xinv


@jax.jit
def _update_svd(
    op: Array,
    size: int,
    basis: Array
) -> tuple[bool, Array]:
    """Update the basis if the basis + [op] matrix is full rank."""
    new_basis = basis.at[size].set(op)
    svals = jnp.linalg.svdvals(new_basis.reshape(basis.shape[:1] + (-1,)))
    rank = jnp.sum(jnp.logical_not(jnp.isclose(svals, 0.)).astype(int))
    fullrank = jnp.equal(rank, size + 1)
    basis = jax.lax.select(fullrank, new_basis, basis)
    return fullrank, basis


@partial(jax.jit, static_argnames=['algorithm'])
def _update_basis(
    op: Array,
    basis_size: int,
    basis: Array,
    aux: list,
    algorithm: Algorithms
) -> tuple[int, Array, list]:
    match algorithm:
        case Algorithms.GS_DIRECT:
            update = _update_gs_direct
        case Algorithms.GRAM_SCHMIDT:
            update = _update_gram_schmidt
        case Algorithms.MATRIX_INV:
            update = _update_matrix_inv
        case Algorithms.SVD:
            update = _update_svd

    def update_with_norm(_op, _basis_size, _basis, *_aux):
        return update(normalize(_op), _basis_size, _basis, *_aux)

    def no_update(_, _basis_size, _basis, *_aux):
        return False, _basis, *_aux

    # If non-null, call the updater function
    updated, basis, *aux = jax.lax.cond(
        jnp.allclose(op, 0.),
        no_update,
        update_with_norm,
        op, basis_size, basis, *aux
    )
    basis_size = jax.lax.select(updated, basis_size + 1, basis_size)
    return basis_size, basis, list(aux)


def get_loop_body(algorithm: Algorithms, print_every: Optional[int] = None):
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

    @jax.jit
    def loop_body(val: tuple) -> tuple:
        """Compute the commutator and update the basis with orthogonal components."""
        idx_gen, idx_op, basis_size, generators, basis, aux = val

        if print_every > 0:
            icomm = idx_op * generators.shape[0] + idx_gen
            jax.lax.cond(
                jnp.equal(icomm % print_every, 0),
                lambda: jax.debug.callback(callback, idx_gen, idx_op, basis_size, generators),
                lambda: None
            )

        basis_size, basis, aux = _update_basis(commutator(generators[idx_gen], basis[idx_op]),
                                               basis_size, basis, aux, algorithm=algorithm)

        # Compute the next indices
        idx_gen = (idx_gen + 1) % generators.shape[0]
        idx_op = jax.lax.select(jnp.equal(idx_gen, 0), idx_op + 1, idx_op)

        return idx_gen, idx_op, basis_size, generators, basis, aux

    return loop_body


def _lie_basis(
    ops: Sequence[np.ndarray],
    *,
    algorithm: Algorithms = Algorithms.GS_DIRECT
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    ops = jnp.array(ops)

    if ops.shape[0] == 0:
        raise ValueError('Cannot determine the basis of null space')

    max_size = ((ops.shape[0] - 1) // BASIS_ALLOC_UNIT + 1) * BASIS_ALLOC_UNIT
    first_op = normalize(ops[0])

    # Set the aux arrays
    match algorithm:
        case Algorithms.GRAM_SCHMIDT:
            # Orthonormal basis
            aux = [_zeros_with_entries(max_size, first_op[None, ...])]
        case Algorithms.MATRIX_INV:
            # X matrix and inverse
            xmat = jnp.eye(max_size, dtype=np.complex128)
            xinv = xmat
            aux = [xmat, xinv]
        case _:
            aux = []

    # Initialize a list of normalized generators
    basis = _zeros_with_entries(ops.shape[0], first_op[None, ...])
    size, basis, aux = jax.lax.fori_loop(
        1, ops.shape[0],
        lambda iop, val: _update_basis(ops[iop], val[0], val[1], val[2], algorithm=algorithm),
        (1, basis, aux)
    )
    basis = basis[:size]

    return basis, aux


def _truncate_arrays(
    basis: np.ndarray,
    aux: list,
    size: int,
    *,
    algorithm: Algorithms = Algorithms.GS_DIRECT
):
    """Truncate the basis and auxiliary arrays to the given size."""
    basis = np.asarray(basis[:size])
    match algorithm:
        case Algorithms.GRAM_SCHMIDT:
            aux[0] = np.asarray(aux[0][:size])
        case Algorithms.MATRIX_INV:
            aux[0] = np.asarray(aux[0][:size, :size])
            aux[1] = np.asarray(aux[1][:size, :size])

    return basis, aux


@partial(jax.jit, static_argnames=['max_size', 'algorithm'])
def _resize_arrays(
    basis: Array,
    aux: list,
    max_size: int,
    *,
    algorithm: Algorithms = Algorithms.GS_DIRECT
):
    """Expand the arrays to a new size."""
    old_size = basis.shape[0]
    new_shape = (max_size,) + basis.shape[1:]
    basis = jnp.zeros(new_shape, dtype=basis.dtype).at[:old_size].set(basis)

    match algorithm:
        case Algorithms.GRAM_SCHMIDT:
            aux = [jnp.zeros(new_shape, dtype=basis.dtype).at[:old_size].set(aux[0])]
        case Algorithms.MATRIX_INV:
            aux = [jnp.eye(max_size, dtype=mat.dtype).at[:old_size, :old_size].set(mat)
                   for mat in aux]

    return basis, aux


def lie_basis(
    ops: Sequence[np.ndarray],
    *,
    algorithm: Algorithms = Algorithms.GS_DIRECT,
    return_aux: bool = False
) -> np.ndarray | tuple[np.ndarray, list]:
    """Identify a basis for the linear space spanned by ops.

    Args:
        ops: Lie algebra elements whose span to calculate the basis for.
        algorithm: Algorithm to use for linear independence check and basis update.
        return_aux: Whether to return the auxiliary objects together with the main output.

    Returns:
        A list of linearly independent ops. If return_aux=True, a list of auxiliary objects
        dependent on the algorithm is returned in addition.
    """
    basis, aux = _lie_basis(ops, algorithm=algorithm)
    if return_aux:
        basis, aux = _truncate_arrays(basis, aux, basis.shape[0], algorithm=algorithm)
        return basis, aux
    return basis


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
        algorithm: Algorithm to use for linear independence check and basis update.
        return_aux: Whether to return the auxiliary objects together with the main output.

    Returns:
        A basis of the Lie algebra spanned by the nested commutators of the generators. If
        return_aux=True, a list of auxiliary objects is returned in addition.
    """
    generators, aux = _lie_basis(generators, algorithm=algorithm)
    LOG.info('Number of independent generators: %d', generators.shape[0])
    if generators.shape[0] <= 1:
        if return_aux:
            return generators, aux
        return generators

    # Get the main loop function for the algorithm
    main_loop_body = get_loop_body(algorithm, print_every=print_every)
    max_dim = max_dim or generators.shape[-1] ** 2 - 1

    # Initialize the basis
    basis_size = generators.shape[0]
    max_size = ((basis_size - 1) // BASIS_ALLOC_UNIT + 1) * BASIS_ALLOC_UNIT
    basis = _zeros_with_entries(max_size, generators)

    # First compute the commutators among the generators
    idx_gens, idx_ops = np.array(
        [[ig, io] for io in range(generators.shape[0]) for ig in range(io)]
    ).T
    basis_size, generators, basis, aux = jax.lax.fori_loop(
        0, len(idx_gens),
        lambda j, val: val[:2] + main_loop_body((val[0][j], val[1][j], *val[2:]))[2:],
        (idx_gens, idx_ops, basis_size, generators, basis, aux)
    )[2:]

    # This would be stupid but possible
    if basis_size >= max_dim:
        basis, aux = _truncate_arrays(basis, aux, basis_size, algorithm=algorithm)
        if return_aux:
            return basis, aux
        return basis

    # Main loop: generate nested commutators
    idx_gen = 0
    idx_op = generators.shape[0]
    while True:  # Outer loop to handle memory reallocation
        LOG.info('Current Lie algebra dimension: %d', basis_size)
        main_loop_start = time.time()
        # Main (inner) loop: iteratively compute the next commutator and update the basis based on
        # the current one
        idx_gen, idx_op, new_size, generators, basis, aux = jax.lax.while_loop(
            lambda val: jnp.logical_not(
                (val[1] == val[2])  # idx_op == new_size -> commutator exhausted
                | (val[2] == max_dim)  # new_size == max_dim -> reached max dim
                | (val[2] == val[4].shape[0])  # new_size == array size -> need reallocation
            ),
            main_loop_body,
            (idx_gen, idx_op, basis_size, generators, basis, aux)
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
        basis, aux = _resize_arrays(basis, aux, max_size, algorithm=algorithm)

    basis, aux = _truncate_arrays(basis, aux, basis_size, algorithm=algorithm)
    if return_aux:
        return basis, aux
    return basis
