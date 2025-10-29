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
from fastdla.linalg.matrix_ops import commutator, innerprod, normalize, orthogonalize
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
def _update_gram_schmidt(
    op: Array,
    basis: Array,
    size: int
) -> tuple[Array, int]:
    """Update the basis if op has an orthogonal component."""
    has_orthcomp, orth = _has_orthcomp(op, basis)
    return jax.lax.cond(
        has_orthcomp,
        lambda _orth, _basis, _size: (_basis.at[_size].set(_orth), _size + 1),
        lambda _, _basis, _size: (_basis, _size),
        orth, basis, size
    )


@jax.jit
def _update_gs_commlist(
    op: Array,
    basis: Array,
    size: int,
    nested_commutators: Array
) -> tuple[Array, int, Array]:
    """Update the basis and the nested commutators list if op has an orthogonal component."""
    has_orthcomp, orth = _has_orthcomp(op, basis)
    return jax.lax.cond(
        has_orthcomp,
        lambda _orth, _basis, _size, _comm, _comms: (
            _basis.at[_size].set(_orth),
            _size + 1,
            _comms.at[_size].set(_comm)
        ),
        lambda a, _basis, _size, b, _comms: (_basis, _size, _comms),
        orth, basis, size, op, nested_commutators
    )


@jax.jit
def _update_matrix_inv(
    op: Array,
    basis: Array,
    size: int,
    xmat: Array,
    xinv: Array
) -> tuple[Array, int, Array, Array]:
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
        return _basis, _size + 1, _xmat, jnp.linalg.inv(_xmat)

    return jax.lax.cond(
        is_independent,
        _update,
        lambda a, b, _basis, _size, _xmat, _xinv: (_basis, _size, _xmat, _xinv),
        op, new_xcol, basis, size, xmat, xinv
    )


@jax.jit
def _update_svd(
    op: Array,
    basis: Array,
    size: int
) -> tuple[Array, int]:
    """Update the basis if the basis + [op] matrix is full rank."""
    new_basis = basis.at[size].set(op)
    svals = jnp.linalg.svdvals(new_basis.reshape(basis.shape[:1] + (-1,)))
    rank = jnp.sum(jnp.logical_not(jnp.isclose(svals, 0.)).astype(int))
    fullrank = jnp.equal(rank, size + 1)
    basis = jax.lax.select(fullrank, new_basis, basis)
    return basis, size + fullrank.astype(int)


@partial(jax.jit, static_argnames=['algorithm'])
def _compute_commutator(
    idx_gen: int,
    idx_op: int,
    generators: Array,
    basis: Array,
    *aux,
    algorithm: Algorithms
) -> Array:
    match algorithm:
        case Algorithms.GS_DIRECT:
            op1, op2 = generators[idx_gen], basis[idx_op]
        case Algorithms.GRAM_SCHMIDT:
            op1, op2 = aux[0][idx_gen], aux[0][idx_op]
        case Algorithms.MATRIX_INV:
            op1, op2 = generators[idx_gen], basis[idx_op]
        case Algorithms.SVD:
            op1, op2 = generators[idx_gen], basis[idx_op]

    return normalize(commutator(op1, op2))


@partial(jax.jit, static_argnames=['algorithm'])
def _update_basis(
    op: Array,
    basis: Array,
    basis_size: int,
    *aux,
    algorithm: Algorithms
) -> tuple:
    match algorithm:
        case Algorithms.GS_DIRECT:
            update = _update_gram_schmidt
        case Algorithms.GRAM_SCHMIDT:
            update = _update_gs_commlist
        case Algorithms.MATRIX_INV:
            update = _update_matrix_inv
        case Algorithms.SVD:
            update = _update_svd

    def no_update(_, _basis, _basis_size, *_aux):
        return _basis, _basis_size, *_aux

    # If non-null, call the updater function
    return jax.lax.cond(
        jnp.allclose(op, 0.),
        no_update,
        update,
        op, basis, basis_size, *aux
    )


def get_loop_body(algorithm: Algorithms, log_level: Optional[int] = None):
    """Make the compiled loop body function using the given algorithm."""
    if log_level is None:
        log_level = LOG.getEffectiveLevel()

    @jax.jit
    def loop_body(val: tuple) -> tuple:
        """Compute the commutator and update the basis with orthogonal components."""
        idx_gen, idx_op, generators, basis, basis_size, *aux = val

        if log_level <= logging.INFO:
            icomm = idx_op * generators.shape[0] + idx_gen
            jax.lax.cond(
                jnp.equal(icomm % 2000, 0),
                lambda: jax.debug.print('Basis size {size}; {icomm}th/{total} commutator'
                                        ' [g[{idx_gen}], b[{idx_new}]]',
                                        size=basis_size, icomm=icomm,
                                        total=basis_size * generators.shape[0],
                                        idx_gen=idx_gen, idx_new=idx_op),
                lambda: None
            )

        comm = _compute_commutator(idx_gen, idx_op, generators, basis, *aux, algorithm=algorithm)
        basis, basis_size, *aux = _update_basis(comm, basis, basis_size, *aux, algorithm=algorithm)

        # Compute the next indices
        idx_gen = (idx_gen + 1) % generators.shape[0]
        idx_op = jax.lax.select(jnp.equal(idx_gen, 0), idx_op + 1, idx_op)

        return idx_gen, idx_op, generators, basis, basis_size, *aux

    return loop_body


def _resize_basis(
    basis: Array,
    size: int,
    max_size: int
) -> tuple[Array]:
    new_shape = (max_size,) + basis.shape[1:]
    return (jnp.resize(basis, new_shape).at[size:].set(0.),)


def _resize_basis_and_x(
    basis: Array,
    size: int,
    max_size: int,
    xmat: Array,
    xinv: Array
) -> tuple[Array, Array, Array]:
    basis, = _resize_basis(basis, size, max_size)
    xmat = jnp.eye(max_size, dtype=xmat.dtype).at[:size, :size].set(xmat)
    xinv = jnp.linalg.inv(xmat)
    return (basis, xmat, xinv)


def _resize_basis_and_commlist(
    basis: Array,
    size: int,
    max_size: int,
    nested_commutators: Array
) -> tuple[Array, Array]:
    basis, = _resize_basis(basis, size, max_size)
    nested_commutators = jnp.resize(nested_commutators, basis.shape).at[size:].set(0.)
    return (basis, nested_commutators)


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

    # Set the aux arrays and resize function
    match algorithm:
        case Algorithms.GRAM_SCHMIDT:
            # Unorthogonalized commutator list
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
    size = 1
    for op in ops[1:]:
        basis, size, *aux = _update_basis(normalize(op), basis, size, *aux, algorithm=algorithm)
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
    basis, aux = _truncate_arrays(basis, aux, basis.shape[0], algorithm=algorithm)
    if return_aux:
        return basis, aux
    return basis


def lie_closure(
    generators: Sequence[np.ndarray],
    *,
    max_dim: Optional[int] = None,
    algorithm: Algorithms = Algorithms.GS_DIRECT,
    return_aux: bool = False
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

    # Fix the main loop function with algorithm
    main_loop_body = get_loop_body(algorithm)
    max_dim = max_dim or generators.shape[-1] ** 2 - 1

    # Set the resize function
    match algorithm:
        case Algorithms.GRAM_SCHMIDT:
            resize = _resize_basis_and_commlist
        case Algorithms.MATRIX_INV:
            resize = _resize_basis_and_x
        case _:
            resize = _resize_basis

    # Initialize the basis
    basis_size = generators.shape[0]
    max_size = ((basis_size - 1) // BASIS_ALLOC_UNIT + 1) * BASIS_ALLOC_UNIT
    basis = _zeros_with_entries(max_size, generators)

    # First compute the commutators among the generators
    idx_gens, idx_ops = np.array(
        [[ig, io] for io in range(generators.shape[0]) for ig in range(io)]
    ).T
    generators, basis, basis_size, *aux = jax.lax.fori_loop(
        0, len(idx_gens),
        lambda j, val: val[:2] + main_loop_body((val[0][j], val[1][j], *val[2:]))[2:],
        (idx_gens, idx_ops, generators, basis, basis_size, *aux)
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
        idx_gen, idx_op, generators, basis, new_size, *aux = jax.lax.while_loop(
            lambda val: jnp.logical_not(
                (val[1] == val[4])  # idx_op == new_size -> commutator exhausted
                | (val[4] == max_dim)  # new_size == max_dim -> reached max dim
                | (val[4] == basis.shape[0])  # new_size == array size -> need reallocation
            ),
            main_loop_body,
            (idx_gen, idx_op, generators, basis, basis_size, *aux)
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
        basis, *aux = resize(basis, basis_size, max_size, *aux)

    basis, aux = _truncate_arrays(basis, aux, basis_size, algorithm=algorithm)
    if return_aux:
        return basis, aux
    return basis
