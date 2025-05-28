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
def _orthogonalize(
    new_op: Array,
    basis: Array
) -> Array:
    return new_op - jnp.tensordot(_innerprod(basis, new_op), basis, [[0], [0]])


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


@jax.jit
def _if_independent_update(
    op: Array,
    basis: Array,
    size: int
) -> tuple[Array, int]:
    def _continue(_, _basis, _size):
        return _basis, _size

    def _update(_orth, _basis, _size):
        _basis = _basis.at[_size].set(_orth)
        return _basis, _size + 1

    orth = _orthogonalize(_normalize(_orthogonalize(op, basis)), basis)
    norm = jnp.sqrt(_innerprod(orth, orth))

    return jax.lax.cond(
        jnp.isclose(norm, 1., rtol=1.e-5),
        _update,
        _continue,
        orth / norm, basis, size
    )


@partial(jax.jit, static_argnames=['keep_original', 'log_level'])
def _main_loop_body(val, keep_original=False, log_level=0):
    """Compute the commutator and update the basis with orthogonal components."""
    if keep_original:
        idx1, idx2, basis, basis_size, nested_commutators = val
        source = nested_commutators
    else:
        idx1, idx2, basis, basis_size = val
        source = basis

    if log_level <= logging.INFO:
        icomm = (idx1 * (idx1 + 1)) // 2 + idx2
        jax.lax.cond(
            jnp.equal(icomm % 2000, 0),
            lambda: jax.debug.print('Basis size {size}; {icomm}th commutator'
                                    ' [b[{idx1}], b[{idx2}]]',
                                    size=basis_size, icomm=icomm, idx1=idx1, idx2=idx2),
            lambda: None
        )

    # Commutator
    comm = _commutator_norm(source[idx1], source[idx2])
    # If non-null, call the updater function
    basis, new_basis_size = jax.lax.cond(
        jnp.allclose(comm, 0.),
        lambda _comm, _basis, _size: (_basis, _size),
        _if_independent_update,
        comm, basis, basis_size
    )
    if keep_original:
        nested_commutators = jax.lax.cond(
            new_basis_size != basis_size,
            lambda _comm, _list, _pos: _list.at[_pos].set(_comm),
            lambda _comm, _list, _pos: _list,
            comm, nested_commutators, basis_size
        )
    # Compute the next indices
    next_idx1, next_idx2 = jax.lax.cond(
        jnp.equal(idx2 + 1, idx1),
        lambda: (idx1 + 1, 0),
        lambda: (idx1, idx2 + 1)
    )

    if keep_original:
        return next_idx1, next_idx2, basis, new_basis_size, nested_commutators
    return next_idx1, next_idx2, basis, new_basis_size


def lie_closure(
    generators: Sequence[np.ndarray],
    *,
    keep_original: bool = False,
    max_dim: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Compute the Lie closure of given generators.

    Args:
        generators: Lie algebra elements to compute the closure from.
        keep_original: Whether to keep the original (normalized) generator elements. If False, only
            orthonormalized Lie algebra elements are kept in memory to speed up the calculation.
        max_dim: Cutoff for the dimension of the Lie closure. If set, the algorithm may be halted
            before a full closure is obtained.

    Returns:
        A list of linearly independent nested commutators and the orthonormal basis if
        keep_original=True, otherwise only the orthonormal basis.
    """
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

    if keep_original:
        nested_commutators = np.array(basis)

    for op in generators[1:]:
        op = _normalize(op)
        basis, new_basis_size = _if_independent_update(op, basis, basis_size)
        if keep_original and new_basis_size != basis_size:
            nested_commutators[basis_size] = op  # pylint: disable=possibly-used-before-assignment
        basis_size = new_basis_size

    main_loop_body = partial(_main_loop_body,
                             keep_original=keep_original, log_level=LOG.getEffectiveLevel())

    if basis_size >= max_dim:
        basis = np.asarray(basis[:max_dim])
        if keep_original:
            return nested_commutators[:max_dim], basis
        return basis

    if keep_original:
        nested_commutators = jnp.array(nested_commutators)

    idx1, idx2 = 1, 0
    while True:  # Outer loop to handle memory reallocation
        LOG.info('Current Lie algebra dimension: %d', basis_size)
        main_loop_start = time.time()
        # Main (inner) loop: iteratively compute the next commutator and update the basis based on
        # the current one
        loop_val = (idx1, idx2, basis, basis_size)
        if keep_original:
            loop_val += (nested_commutators,)

        loop_val = jax.lax.while_loop(
            lambda val: jnp.logical_not(
                (val[0] == val[3])  # idx1 == new_size -> commutator exhausted
                | (val[3] == max_dim)  # new_size == max_dim -> reached max dim
                | (val[3] == basis.shape[0])  # new_size == array size -> need reallocation
            ),
            main_loop_body,
            loop_val
        )

        idx1, idx2, basis, new_size = loop_val[:4]
        if keep_original:
            nested_commutators = loop_val[-1]

        LOG.info('Found %d new ops in %.2fs',
                 new_size - basis_size, time.time() - main_loop_start)

        basis_size = new_size

        if idx1 == basis_size or basis_size == max_dim:
            # Computed all commutators
            break

        # Need to resize basis and xmat
        LOG.debug('Resizing basis array to %d', max_size + BASIS_ALLOC_UNIT)

        max_size += BASIS_ALLOC_UNIT
        new_shape = (max_size,) + basis.shape[1:]
        basis = jnp.resize(basis, new_shape).at[basis_size:].set(0.)
        if keep_original:
            nested_commutators = jnp.resize(nested_commutators, new_shape).at[basis_size:].set(0.)

    if keep_original:
        return np.asarray(nested_commutators[:basis_size]), np.asarray(basis[:basis_size])
    return np.asarray(basis[:basis_size])
