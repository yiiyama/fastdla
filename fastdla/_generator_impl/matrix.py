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
def _innerprod(op1: Array, op2: Array):
    return jnp.einsum('ij,ij->', op1.conjugate(), op2) / op1.shape[0]


@jax.jit
def _normalize(op: Array):
    norm = jnp.sqrt(_innerprod(op, op))
    return jax.lax.cond(
        jnp.isclose(norm, 0.),
        lambda: jnp.zeros_like(op),
        lambda: op / norm
    )


@jax.jit
def _commutator(op1, op2):
    return op1 @ op2 - op2 @ op1


@jax.jit
def _commutator_norm(op1: Array, op2: Array):
    return _normalize(_commutator(op1, op2))


@jax.jit
def _compute_xmatrix(basis: Array) -> Array:
    xmat = jnp.einsum('ijk,ljk->il', basis.conjugate(), basis)
    idx = np.arange(basis.shape[0])
    return xmat.at[idx, idx].set(1.)


@jax.jit
def _extend_xmatrix(xmat: Array, basis: Array, idx: int):
    new_col = jnp.einsum('ijk,jk->i', basis.conjugate(), basis[idx])
    return xmat.at[:, idx].set(new_col).at[idx, :].set(new_col.conjugate())


@jax.jit
def _linear_independence(
    new_op: Array,
    basis: Array,
    xinv: Array
):
    def _residual(_new_op, _basis, _xmat_inv, _pidag_q):
        # Residual calculation: subtract Pi*ai from Q directly
        a_proj = _xmat_inv @ _pidag_q
        residual = _new_op - jnp.sum(_basis * a_proj[:, None, None], axis=0)
        return jnp.logical_not(jnp.allclose(residual, 0.))

    # Compute the Π†Q vector
    pidag_q = jnp.einsum('ijk,jk->i', basis.conjugate(), new_op)
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


@partial(jax.jit, static_argnames=['verbosity'])
def _main_loop_body(val, verbosity=0):
    idx1, idx2, comm, basis, size, xmat, xinv = val

    if verbosity > 1:
        icomm = (idx1 * (idx1 + 1)) // 2 + idx2
        jax.lax.cond(
            jnp.equal(icomm % 2000, 0),
            lambda: jax.debug.print('Basis size {size}; {icomm}th commutator'
                                    ' [b[{idx1}], b[{idx2}]]',
                                    size=size, icomm=icomm, idx1=idx1, idx2=idx2),
            lambda: None
        )

    def _continue(_, _basis, _size, _xmat, _xmat_inv):
        return _basis, _size, _xmat, _xmat_inv

    def _update(_comm, _basis, _size, _xmat, _xmat_inv):
        _basis = _basis.at[_size].set(_comm)
        _xmat = _extend_xmatrix(_xmat, _basis, _size)
        _xmat_inv = jnp.linalg.inv(_xmat)
        return _basis, _size + 1, _xmat, _xmat_inv

    def _if_independent_update(_comm, _basis, _size, _xmat, _xmat_inv):
        return jax.lax.cond(
            _linear_independence(_comm, _basis, _xmat_inv),
            _update,
            _continue,
            _comm, _basis, _size, _xmat, _xmat_inv
        )

    next_idx1, next_idx2 = jax.lax.cond(
        jnp.equal(idx2 + 1, idx1),
        lambda: (idx1 + 1, 0),
        lambda: (idx1, idx2 + 1)
    )
    # Compute the next commutator in parallel to the independence of the current one
    next_comm = jax.lax.cond(
        jnp.equal(next_idx1, basis.shape[0]),
        lambda: jnp.empty_like(comm),
        lambda: _commutator_norm(basis[next_idx1], basis[next_idx2])
    )
    basis, size, xmat, xinv = jax.lax.cond(
        jnp.allclose(comm, 0.),
        _continue,
        _if_independent_update,
        comm, basis, size, xmat, xinv
    )
    return (next_idx1, next_idx2, next_comm, basis, size, xmat, xinv)


def lie_closure(
    generators: Sequence[np.ndarray],
    *,
    keep_original: bool = True,
    max_dim: Optional[int] = None,
    verbosity: int = 0,
) -> list[np.ndarray]:
    if (size := len(generators)) == 0:
        return []

    max_size = ((size - 1) // XMAT_ALLOC_UNIT + 1) * XMAT_ALLOC_UNIT
    basis = jnp.zeros((max_size,) + generators.shape[1:], dtype=generators.dtype)
    for iop, op in enumerate(generators):
        basis.at[iop].update(_normalize(jnp.asarray(op)))

    xmat = _compute_xmatrix(basis)
    xinv = jnp.linalg.inv(xmat[:size, :size])
    idx1, idx2 = 1, 0

    main_loop_body = partial(_main_loop_body, verbosity=verbosity)

    while True:
        main_loop_start = time.time()

        comm = _commutator_norm(basis[idx1], basis[idx2])
        idx1, idx2, _, basis, new_size, xmat, _ = jax.lax.while_loop(
            lambda val: jnp.logical_and(
                jnp.not_equal(val[0], val[4]),
                jnp.not_equal(val[4], basis.shape[0])
            ),
            main_loop_body,
            (idx1, idx2, comm, basis, size, xmat, xinv)
        )

        if verbosity > 0:
            print(f'Found {new_size - size} new ops in {time.time() - main_loop_start:.2f}s')

        size = new_size

        if idx1 != size:
            # Need to resize basis and xmat
            if verbosity > 0:
                print(f'Resizing basis array to {max_size + XMAT_ALLOC_UNIT}')

            max_size += XMAT_ALLOC_UNIT
            basis = jnp.resize(basis, (max_size,) + basis.shape[1:]).at[size:].set(0.)
            xmat = jnp.eye(max_size, dtype=xmat.dtype).at[:size, :size].set(xmat)
            xinv = jnp.linalg.inv(xmat[:size, :size])
            continue

        if max_dim is not None and size >= max_dim:
            basis = basis[:max_dim]

        break

    return np.asarray(basis[:size])
