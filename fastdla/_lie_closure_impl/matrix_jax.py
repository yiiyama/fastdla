# pylint: disable=unused-argument
"""Implementation of the Lie closure generator using JAX matrices."""
from collections.abc import Callable, Sequence
from functools import partial
import logging
from typing import Optional
import time
import numpy as np
import jax
from jax import Array
import jax.numpy as jnp

LOG = logging.getLogger(__name__)
BASIS_ALLOC_UNIT = 1024


@jax.jit
def _innerprod(op1: Array, op2: Array) -> complex:
    """Inner product between two (stacked) matrices defined by Tr(A†B)/d."""
    return jnp.tensordot(op1.conjugate(), op2, [[-2, -1], [-2, -1]]) / op1.shape[-1]


@jax.jit
def _normalize(op: Array) -> Array:
    """Normalize a matrix."""
    norm = jnp.sqrt(_innerprod(op, op))
    return jax.lax.cond(
        jnp.isclose(norm, 0.),
        lambda _op, _norm: jnp.zeros_like(_op),
        lambda _op, _norm: _op / _norm,
        op, norm
    )


@jax.jit
def _commutator_norm(op1: Array, op2: Array) -> Array:
    """Normalized commutator."""
    return _normalize(op1 @ op2 - op2 @ op1)


@jax.jit
def _commutator_from_basis(idx1: int, idx2: int, basis: Array) -> Array:
    return _commutator_norm(basis[idx1], basis[idx2])


@jax.jit
def _commutator_from_commlist(idx1: int, idx2: int, _: Array, commlist: Array) -> Array:
    return _commutator_norm(commlist[idx1], commlist[idx2])


@jax.jit
def _commutator_from_basis_2aux(
    idx1: int,
    idx2: int,
    basis: Array,
    aux1: Array,
    aux2: Array
) -> Array:
    return _commutator_from_basis(idx1, idx2, basis)


@jax.jit
def _orthogonalize(
    new_op: Array,
    basis: Array
) -> Array:
    return new_op - jnp.tensordot(_innerprod(basis, new_op), basis, [[0], [0]])


@jax.jit
def _has_orthcomp(op: Array, basis: Array) -> tuple[bool, Array, float]:
    orth = _orthogonalize(_normalize(_orthogonalize(op, basis)), basis)
    norm = jnp.sqrt(_innerprod(orth, orth))
    return jnp.isclose(norm, 1., rtol=1.e-5), orth, norm


@jax.jit
def _if_orthogonal_update(
    op: Array,
    basis: Array,
    size: int
) -> tuple[Array, int]:
    """Update the basis if op has an orthogonal component."""
    has_orthcomp, orth, norm = _has_orthcomp(op, basis)
    return jax.lax.cond(
        has_orthcomp,
        lambda _orth, _basis, _size: (_basis.at[_size].set(_orth), _size + 1),
        lambda _, _basis, _size: (_basis, _size),
        orth / norm, basis, size
    )


@jax.jit
def _if_orthogonal_update_suppl(
    op: Array,
    basis: Array,
    size: int,
    nested_commutators: Array
) -> tuple[Array, int, Array]:
    """Update the basis and the nested commutators list if op has an orthogonal component."""
    has_orthcomp, orth, norm = _has_orthcomp(op, basis)

    return jax.lax.cond(
        has_orthcomp,
        lambda _orth, _basis, _size, _comm, _comms: (
            _basis.at[_size].set(_orth),
            _size + 1,
            _comms.at[_size].set(_comm)
        ),
        lambda _orth, _basis, _size, _comm, _comms: (_basis, _size, _comms),
        orth / norm, basis, size, op, nested_commutators
    )


@jax.jit
def _if_fullrank_update(
    op: Array,
    basis: Array,
    size: int
) -> tuple[Array, int]:
    """Update the basis if the basis + [op] matrix is full rank."""
    new_basis = basis.at[size].set(op)
    svals = jnp.linalg.svdvals(new_basis.reshape(basis.shape[:1] + (-1,)))
    rank = jnp.sum(jnp.logical_not(jnp.isclose(svals, 0.)).astype(int))

    return jax.lax.cond(
        jnp.equal(rank, size + 1),
        lambda _basis, _new_basis, _size: (_new_basis, _size + 1),
        lambda _basis, _new_basis, _size: (_basis, _size),
        basis, new_basis, size
    )


@jax.jit
def _is_independent(
    new_op: Array,
    basis: Array,
    xinv: Array
) -> tuple[bool, Array]:
    """Check linear independence of a matrix with respect to the basis."""
    def _residual(_new_op, _basis, _xinv, _pidag_q):
        # Residual calculation: subtract Pi*ai from Q directly
        a_proj = _xinv @ _pidag_q
        residual = _new_op - jnp.sum(_basis * a_proj[:, None, None], axis=0)
        return jnp.logical_not(jnp.allclose(residual, 0.)), _pidag_q

    # Compute the Π†Q vector
    pidag_q = _innerprod(basis, new_op)
    # If pidag_q is non-null, compute the residual
    return jax.lax.cond(
        jnp.allclose(pidag_q, 0.),
        lambda a, b, c, _pidag_q: (True, _pidag_q),
        _residual,
        new_op, basis, xinv, pidag_q
    )


@jax.jit
def _if_independent_update(
    op: Array,
    basis: Array,
    size: int,
    xmat: Array,
    xinv: Array
) -> tuple[Array, int, Array, Array]:
    """Update the basis and the X matrix with op if it is independent."""
    def _update(_op, _new_xcol, _basis, _size, _xmat, _):
        _basis = _basis.at[_size].set(_op)
        _xmat = _xmat.at[:, _size].set(_new_xcol).at[size, :].set(_new_xcol.conjugate())
        _xmat = _xmat.at[_size, _size].set(1.)
        return _basis, _size + 1, _xmat, jnp.linalg.inv(_xmat)

    is_independent, new_xcol = _is_independent(op, basis, xinv)

    return jax.lax.cond(
        is_independent,
        _update,
        lambda _op, _new_xcol, _basis, _size, _xmat, _xinv: (_basis, _size, _xmat, _xinv),
        op, new_xcol, basis, size, xmat, xinv
    )


@partial(jax.jit, static_argnames=['log_level', 'commutator', 'updater'])
def _main_loop_body(
    val: tuple[int, int, Array, int],
    log_level: int = 0,
    commutator: Callable = _commutator_from_basis,
    updater: Callable = _if_orthogonal_update
) -> tuple[int, int, Array, int]:
    """Compute the commutator and update the basis with orthogonal components."""
    def _continue(_, _basis, _basis_size, *_aux):
        return _basis, _basis_size, *_aux

    idx1, idx2, basis, basis_size, *aux = val

    if log_level <= logging.INFO:
        icomm = (idx1 * (idx1 + 1)) // 2 + idx2
        jax.lax.cond(
            jnp.equal(icomm % 2000, 0),
            lambda: jax.debug.print('Basis size {size}; {icomm}th/{total} commutator'
                                    ' [b[{idx1}], b[{idx2}]]',
                                    size=basis_size, icomm=icomm,
                                    total=(basis_size - 1) * basis_size // 2, idx1=idx1, idx2=idx2),
            lambda: None
        )

    # Commutator
    comm = commutator(idx1, idx2, basis, *aux)
    # If non-null, call the updater function
    basis, basis_size, *aux = jax.lax.cond(
        jnp.allclose(comm, 0.),
        _continue,
        updater,
        comm, basis, basis_size, *aux
    )
    # Compute the next indices
    idx1, idx2 = jax.lax.cond(
        jnp.equal(idx2 + 1, idx1),
        lambda: (idx1 + 1, 0),
        lambda: (idx1, idx2 + 1)
    )

    return idx1, idx2, basis, basis_size, *aux


def _resize_basis(
    basis: Array,
    size: int,
    max_size: int
) -> tuple[Array]:
    new_shape = (max_size,) + basis.shape[1:]
    return jnp.resize(basis, new_shape).at[size:].set(0.),


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
    return basis, xmat, xinv


def _resize_basis_and_commlist(
    basis: Array,
    size: int,
    max_size: int,
    nested_commutators: Array
) -> tuple[Array, Array]:
    basis, = _resize_basis(basis, size, max_size)
    nested_commutators = jnp.resize(nested_commutators, basis.shape).at[size:].set(0.)
    return basis, nested_commutators


def orthogonalize(
    new_op: Array,
    basis: Array,
    normalize: bool = True
) -> Array:
    """Subtract the subspace projection of an algebra element from itself.

    See the docstring of generator.orthogonalize for details of the algorithm.

    Args:
        op: Lie algebra element Q to check the linear independence of.
        projector:

    Returns:
        True if Q is linearly independent from all elements of the basis.
    """
    orth = _orthogonalize(new_op, basis)
    if normalize:
        orth = _normalize(orth)
    return orth


def lie_closure(
    generators: Sequence[np.ndarray],
    *,
    keep_original: bool = False,
    max_dim: Optional[int] = None,
    algorithm: str = 'default'
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Compute the Lie closure of given generators.

    Args:
        generators: Lie algebra elements to compute the closure from.
        keep_original: Whether to keep the original (normalized) generator elements. If False, only
            orthonormalized Lie algebra elements are kept in memory to speed up the calculation.
        max_dim: Cutoff for the dimension of the Lie closure. If set, the algorithm may be halted
            before a full closure is obtained.
        algorithm: Choice of linear-independence check method. In general there is no need for
            values other than 'default'; this feature was used for demonstrations in
            arXiv:2506.01120. Options: 'default', 'keep_original', 'matrix_inversion', 'svd'.
            Algorithm='keep_original' is equivalent to 'default' with keep_original=True.

    Returns:
        A list of linearly independent nested commutators and the orthonormal basis if
        keep_original=True, otherwise only the orthonormal basis.
    """
    if algorithm == 'keep_original':
        algorithm = 'default'
        keep_original = True

    if keep_original and algorithm != 'default':
        raise ValueError('keep_original=True is only valid for default algorithm')

    if len(generators) == 0:
        if keep_original:
            return np.array(generators), np.array(generators)
        return np.array(generators)

    max_dim = max_dim or generators[0].shape[-1] ** 2 - 1

    # Allocate the basis array and compute the initial basis
    max_size = ((len(generators) - 1) // BASIS_ALLOC_UNIT + 1) * BASIS_ALLOC_UNIT
    basis = jnp.zeros((max_size,) + generators[0].shape, dtype=generators[0].dtype)
    basis = basis.at[0].set(_normalize(generators[0]))
    basis_size = 1

    commutator = _commutator_from_basis
    resizer = _resize_basis
    aux = []
    if algorithm == 'svd':
        updater = _if_fullrank_update
    elif algorithm == 'matrix_inversion':
        updater = _if_independent_update
        commutator = _commutator_from_basis_2aux
        resizer = _resize_basis_and_x
        xmat = jnp.eye(max_size, dtype=np.complex128)
        xinv = xmat
        aux = [xmat, xinv]
    elif keep_original:
        updater = _if_orthogonal_update_suppl
        commutator = _commutator_from_commlist
        resizer = _resize_basis_and_commlist
        aux = [jnp.array(basis)]
    else:
        updater = _if_orthogonal_update

    for op in generators[1:]:
        op = _normalize(op)
        basis, basis_size, *aux = updater(op, basis, basis_size, *aux)

    main_loop_body = partial(_main_loop_body,
                             log_level=LOG.getEffectiveLevel(),
                             commutator=commutator, updater=updater)

    if basis_size >= max_dim:
        if keep_original:
            return np.asarray(aux[0][:max_dim]), np.asarray(basis[:max_dim])
        return np.asarray(basis[:max_dim])

    idx1, idx2 = 1, 0
    while True:  # Outer loop to handle memory reallocation
        LOG.info('Current Lie algebra dimension: %d', basis_size)
        main_loop_start = time.time()
        # Main (inner) loop: iteratively compute the next commutator and update the basis based on
        # the current one
        idx1, idx2, basis, new_size, *aux = jax.lax.while_loop(
            lambda val: jnp.logical_not(
                (val[0] == val[3])  # idx1 == new_size -> commutator exhausted
                | (val[3] == max_dim)  # new_size == max_dim -> reached max dim
                | (val[3] == basis.shape[0])  # new_size == array size -> need reallocation
            ),
            main_loop_body,
            (idx1, idx2, basis, basis_size, *aux)
        )

        LOG.info('Found %d new ops in %.2fs',
                 new_size - basis_size, time.time() - main_loop_start)

        basis_size = new_size

        if idx1 == basis_size or basis_size == max_dim:
            # Computed all commutators
            break

        # Need to resize basis and xmat
        LOG.debug('Resizing basis array to %d', max_size + BASIS_ALLOC_UNIT)

        max_size += BASIS_ALLOC_UNIT
        basis, *aux = resizer(basis, basis_size, max_size, *aux)

    if keep_original:
        return np.asarray(aux[0][:basis_size]), np.asarray(basis[:basis_size])
    return np.asarray(basis[:basis_size])
