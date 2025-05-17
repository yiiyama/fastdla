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
def _if_independent_update(op: Array, basis: Array, size: int, xmat: Array, xinv: Array):
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


@partial(jax.jit, static_argnames=['verbosity'])
def _main_loop_body(val, verbosity=0):
    """Compute the commutator and update the basis if linearly independent."""
    # Current commutator indices, commutator, current basis and size, current X matrix and inverse
    idx1, idx2, basis, basis_size, xmat, xinv = val

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
    basis, basis_size, xmat, xinv = _if_independent_update(comm, basis, basis_size, xmat, xinv)
    # Indices for the next commutator
    next_idx1, next_idx2 = jax.lax.cond(
        jnp.equal(idx2 + 1, idx1),
        lambda: (idx1 + 1, 0),
        lambda: (idx1, idx2 + 1)
    )

    return next_idx1, next_idx2, basis, basis_size, xmat, xinv


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
        keep_original: Whether to keep the original generator elements. If False, only
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
    xmat = jnp.eye(max_size, dtype=np.complex128)
    basis_size = 0
    for op in generators:
        basis, basis_size, xmat = _update_basis(_normalize(op), basis, basis_size, xmat)
    xinv = jnp.linalg.inv(xmat)

    main_loop_body = partial(_main_loop_body, verbosity=verbosity)
    idx1, idx2 = 1, 0
    while True:  # Outer loop to handle memory reallocation
        main_loop_start = time.time()
        # Main (inner) loop: iteratively compute the next commutator and update the basis based on
        # the current one
        idx1, idx2, basis, new_size, xmat, _ = jax.lax.while_loop(
            lambda val: jnp.logical_and(
                jnp.not_equal(val[0], val[3]),  # idx1 == new_size -> commutator exhausted
                jnp.not_equal(val[3], basis.shape[0])  # new_size == array size -> need reallocation
            ),
            main_loop_body,
            (idx1, idx2, basis, basis_size, xmat, xinv)
        )

        if verbosity > 0:
            print(f'Found {new_size - basis_size} new ops in {time.time() - main_loop_start:.2f}s')

        basis_size = new_size

        if idx1 != basis_size:
            # Need to resize basis and xmat
            if verbosity > 0:
                print(f'Resizing basis array to {max_size + XMAT_ALLOC_UNIT}')

            max_size += XMAT_ALLOC_UNIT
            basis = jnp.resize(basis, (max_size,) + basis.shape[1:]).at[basis_size:].set(0.)
            xmat = jnp.eye(max_size, dtype=xmat.dtype).at[:basis_size, :basis_size].set(xmat)
            xinv = jnp.linalg.inv(xmat)
            continue

        if max_dim is not None and basis_size >= max_dim:
            basis = basis[:max_dim]

        break

    return np.asarray(basis[:basis_size])
