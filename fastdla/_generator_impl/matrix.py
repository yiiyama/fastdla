"""Implementation of the Lie closure generator using JAX matrices."""
from collections.abc import Sequence
from functools import partial
from typing import Optional
import time
import numpy as np
import jax
from jax import Array
import jax.numpy as jnp

XMAT_ALLOC_UNIT = 1024


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
def _linear_independence(
    new_op: Array,
    basis: Array,
    xinv: Array
) -> bool:
    """Check linear independence of a matrix with respect to the basis."""
    def _residual(_new_op, _basis, _xinv, _pidag_q):
        # Residual calculation: subtract Pi*ai from Q directly
        a_proj = _xinv @ _pidag_q
        residual = _new_op - jnp.sum(_basis * a_proj[:, None, None], axis=0)
        return jnp.logical_not(jnp.allclose(residual, 0.))

    # Compute the Π†Q vector
    pidag_q = _innerprod(basis, new_op)
    # If pidag_q is non-null, compute the residual
    return jax.lax.cond(
        jnp.allclose(pidag_q, 0.),
        lambda a, b, c, d: True,
        _residual,
        new_op, basis, xinv, pidag_q
    )


def linear_independence(
    new_op: Array,
    basis: Array,
    xinv: Array
) -> bool:
    """Check if the given operator is linearly independent from all other elements in the basis.

    See the docstring of generator.linear_independence for details of the algorithm.

    Args:
        new_op: Lie algebra element Q to check the linear independence of.
        basis: The basis (list of linearly independent elements) of the Lie Algebra.
        xinv: Inverse of the X matrix.

    Returns:
        True if Q is linearly independent from all elements of the basis.
    """
    return _linear_independence(new_op, basis, xinv)


@jax.jit
def _orthogonalize(
    new_op: Array,
    basis: Array
) -> Array:
    return new_op - jnp.tensordot(_innerprod(basis, new_op), basis, [[0], [0]])


def orthogonalize(
    new_op: Array,
    basis: Array
) -> Array:
    """Subtract the subspace projection of an algebra element from itself.

    See the docstring of generator.orthogonalize for details of the algorithm.

    Args:
        op: Lie algebra element Q to check the linear independence of.
        projector:

    Returns:
        True if Q is linearly independent from all elements of the basis.
    """
    return _orthogonalize(new_op, basis)


@jax.jit
def _update_basis(op: Array, basis: Array, size: Array, xmat: Array) -> tuple[Array, int, Array]:
    """Append a new matrix to the basis and extend the X matrix accordingly.

    Args:
        op: New matrix to be added to the basis.
        basis: Current basis of the Lie algebra.
        size: Current size (dimension) of the basis.
        xmat: Current X matrix.

    Returns:
        The updated basis, new basis size, and the updated X matrix.
    """
    new_col = _innerprod(basis, op)
    xmat = xmat.at[:, size].set(new_col).at[size, :].set(new_col.conjugate())
    xmat = xmat.at[size, size].set(1.)
    basis = basis.at[size].set(op)
    return basis, size + 1, xmat


@jax.jit
def _if_independent_update(
    op: Array,
    basis: Array,
    size: int,
    xmat: Array,
    xinv: Array
) -> tuple[Array, int, Array, Array]:
    """Update the basis and the X matrix with op if it is independent."""
    def _continue(_, _basis, _size, _xmat, _xinv):
        return _basis, _size, _xmat, _xinv

    def _update(_comm, _basis, _size, _xmat, _xinv):
        _basis, _size, _xmat = _update_basis(_comm, _basis, _size, _xmat)
        _xinv = jnp.linalg.inv(_xmat)
        return _basis, _size, _xmat, _xinv

    return jax.lax.cond(
        _linear_independence(op, basis, xinv),
        _update,
        _continue,
        op, basis, size, xmat, xinv
    )


@jax.jit
def _if_orthogonal_update(
    op: Array,
    basis: Array,
    size: int
) -> tuple[Array, int]:
    def _continue(_, _basis, _size):
        return _basis, _size

    def _update(_orth, _basis, _size):
        _basis = _basis.at[_size].set(_normalize(_orth))
        return _basis, _size + 1

    orth = _orthogonalize(op, basis)
    return jax.lax.cond(
        jnp.allclose(orth, 0.),
        _continue,
        _update,
        orth, basis, size
    )


@partial(jax.jit, static_argnames=['updater', 'verbosity'])
def _main_loop_body(val, updater, verbosity=0):
    """Compute the commutator and update the basis with orthogonal components."""
    idx1, idx2, basis, basis_size, aux = val

    if verbosity > 1:
        icomm = (idx1 * (idx1 + 1)) // 2 + idx2
        jax.lax.cond(
            jnp.equal(icomm % 2000, 0),
            lambda: jax.debug.print('Basis size {size}; {icomm}th commutator'
                                    ' [b[{idx1}], b[{idx2}]]',
                                    size=basis_size, icomm=icomm, idx1=idx1, idx2=idx2),
            lambda: None
        )

    # Commutator
    comm = _commutator_norm(basis[idx1], basis[idx2])
    # If the current commutator is independent, update the basis and the X matrix
    result = updater(comm, basis, basis_size, *aux)
    basis, basis_size = result[:2]
    aux = result[2:]
    # Indices for the next commutator
    next_idx1, next_idx2 = jax.lax.cond(
        jnp.equal(idx2 + 1, idx1),
        lambda: (idx1 + 1, 0),
        lambda: (idx1, idx2 + 1)
    )

    return next_idx1, next_idx2, basis, basis_size, aux


def lie_closure(
    generators: Sequence[np.ndarray],
    *,
    keep_original: bool = True,
    max_dim: Optional[int] = None,
    verbosity: int = 0,
) -> list[np.ndarray]:
    """Compute the Lie closure of given generators.

    Args:
        generators: Lie algebra elements to compute the closure from.
        keep_original: Whether to keep the original (normalized) generator elements. If False, only
            orthonormalized Lie algebra elements are kept in memory to speed up the calculation.
        max_dim: Cutoff for the dimension of the Lie closure. If set, the algorithm may be halted
            before a full closure is obtained.
        verbosity: Verbosity level.

    Returns:
        A basis of the Lie closure.
    """
    if len(generators) == 0:
        return np.array([], dtype=np.complex128)

    # Allocate basis and xmat arrays and compute the initial basis and X
    max_size = ((len(generators) - 1) // XMAT_ALLOC_UNIT + 1) * XMAT_ALLOC_UNIT
    basis = jnp.zeros((max_size,) + generators[0].shape, dtype=generators[0].dtype)
    basis = basis.at[0].set(_normalize(generators[0]))
    basis_size = 1

    if keep_original:
        xmat = jnp.eye(max_size, dtype=np.complex128)
        xinv = xmat
        for op in generators[1:]:
            basis, basis_size, xmat, xinv = _if_independent_update(_normalize(op), basis,
                                                                   basis_size, xmat, xinv)

        main_loop_body = partial(_main_loop_body,
                                 updater=_if_independent_update, verbosity=verbosity)
        aux = (xmat, xinv)
    else:
        for op in generators[1:]:
            basis, basis_size = _if_orthogonal_update(op, basis, basis_size)

        main_loop_body = partial(_main_loop_body,
                                 updater=_if_orthogonal_update, verbosity=verbosity)
        aux = ()

    idx1, idx2 = 1, 0
    while True:  # Outer loop to handle memory reallocation
        main_loop_start = time.time()
        # Main (inner) loop: iteratively compute the next commutator and update the basis based on
        # the current one
        idx1, idx2, basis, new_size, aux = jax.lax.while_loop(
            lambda val: jnp.logical_and(
                jnp.not_equal(val[0], val[3]),  # idx1 == new_size -> commutator exhausted
                jnp.not_equal(val[3], basis.shape[0])  # new_size == array size -> need reallocation
            ),
            main_loop_body,
            (idx1, idx2, basis, basis_size, aux)
        )

        if verbosity > 0:
            print(f'Found {new_size - basis_size} new ops in {time.time() - main_loop_start:.2f}s')

        basis_size = new_size

        if idx1 == basis_size:
            # Computed all commutators
            break

        if max_dim is not None and basis_size >= max_dim:
            # Cutting off
            basis = basis[:max_dim]
            break

        # Need to resize basis and xmat
        if verbosity > 0:
            print(f'Resizing basis array to {max_size + XMAT_ALLOC_UNIT}')

        max_size += XMAT_ALLOC_UNIT
        basis = jnp.resize(basis, (max_size,) + basis.shape[1:]).at[basis_size:].set(0.)
        if keep_original:
            xmat = jnp.eye(max_size, dtype=xmat.dtype).at[:basis_size, :basis_size].set(xmat)
            xinv = jnp.linalg.inv(xmat)
            aux = (xmat, xinv)

    return np.asarray(basis[:basis_size])
