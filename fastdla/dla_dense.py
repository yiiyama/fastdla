"""Calculation of DLA based on dense matrices."""
from collections.abc import Sequence
from functools import partial
from typing import Optional
import time
import numpy as np
import jax
from jax import Array
import jax.numpy as jnp


@jax.jit
def _commutator_norm(op1, op2):
    comm = op1 @ op2 - op2 @ op1
    norm = jnp.sqrt(jnp.abs(jnp.trace(comm.conjugate().T @ comm)))
    return jnp.where(jnp.isclose(norm, 0.), jnp.zeros_like(op1), comm / norm)


_vcommutator_norm = jax.vmap(_commutator_norm, in_axes=[0, 0])


@jax.jit
def _compute_xmatrix(basis):
    xmat = jnp.einsum('ijk,ljk->il', basis.conjugate(), basis)
    idx = np.arange(basis.shape[0])
    return xmat.at[idx, idx].set(1.)


def compute_xmatrix(basis: Sequence[Array]) -> Array:
    return _compute_xmatrix(jnp.asarray(basis))


@jax.jit
def _is_independent(new_op, basis, xmat_inv):
    def _residual(_new_op, _basis, _xmat_inv, _pidag_q):
        # Residual calculation: subtract Pi*ai from Q directly
        a_proj = _xmat_inv @ _pidag_q
        residual = _new_op - jnp.sum(_basis * a_proj[:, None, None], axis=0)
        return jnp.logical_not(jnp.allclose(residual, 0.))

    pidag_q = jnp.einsum('ijk,jk->i', basis.conjugate(), new_op)
    return jax.lax.cond(
        jnp.allclose(pidag_q, 0.),
        lambda a, b, c, d: True,
        _residual,
        new_op, basis, xmat_inv, pidag_q
    )


def is_independent(
    new_op: Array,
    basis: Sequence[Array],
    xmat_inv: Optional[Array] = None,
) -> bool:
    """
    Let the known dla basis ops be P0, P1, ..., Pn. The basis_matrix Π is a matrix formed by
    stacking the column vectors {Pi}:
    Π = (P0, P1, ..., Pn).
    If new_op Q is linearly dependent on {Pi}, there is a column vector of coefficients
    a = (a0, a1, ..., an)^T where
    Π a = Q.
    Multiply both sides with Π† and denote X = Π†Π to obtain
    X a = Π† Q.
    Since {Pi} are linearly independent, X must be invertible:
    a = X^{-1} Π† Q.
    Using thus calculated {ai}, we check the residual
    R = Q - Π a
    to determine the linear independence of Q with respect to {Pi}.
    """
    if xmat_inv is None:
        xmat_inv = jnp.linalg.inv(compute_xmatrix(basis))
    return _is_independent(new_op, jnp.asarray(basis), xmat_inv)


@partial(jax.jit, static_argnames=['verbosity'])
def _main_loop_body(val, verbosity=0):
    idx1, idx2, basis, size, xmat_inv = val

    if verbosity > 1:
        icomm = (idx1 * (idx1 + 1)) // 2 + idx2
        jax.lax.cond(
            jnp.equal(icomm % 2000, 0),
            lambda: jax.debug.print('Basis size {size}; {icomm}th commutator'
                                    ' [b[{idx1}], b[{idx2}]]',
                                    size=size, icomm=icomm, idx1=idx1, idx2=idx2),
            lambda: None
        )

    def _continue(_, _basis, _size, _xmat_inv):
        return _basis, _size, _xmat_inv

    def _update(_comm, _basis, _size, _xmat_inv):
        _basis = _basis.at[_size].set(_comm)
        _xmat_inv = jnp.linalg.inv(_compute_xmatrix(_basis))
        return _basis, _size + 1, _xmat_inv

    def _if_independent_update(_comm, _basis, _size, _xmat_inv):
        return jax.lax.cond(
            _is_independent(_comm, _basis, _xmat_inv),
            _update,
            _continue,
            _comm, _basis, _size, _xmat_inv
        )

    comm = _commutator_norm(basis[idx1], basis[idx2])
    return jax.lax.cond(
        jnp.equal(idx2 + 1, idx1),
        lambda: (idx1 + 1, 0),
        lambda: (idx1, idx2 + 1)
    ) + jax.lax.cond(
        jnp.allclose(comm, 0.),
        _continue,
        _if_independent_update,
        comm, basis, size, xmat_inv
    )


def generate_dla(
    generators: Sequence[np.ndarray],
    *,
    size_increment: int = 256,
    max_dim: Optional[int] = None,
    verbosity: int = 0
) -> list[np.ndarray]:
    if (size := len(generators)) == 0:
        return []

    max_size = size_increment
    generators = jnp.asarray(generators)
    basis = jnp.resize(generators, (max_size,) + generators.shape[1:]).at[size:].set(0.)
    xmat_inv = jnp.linalg.inv(_compute_xmatrix(basis))
    idx1, idx2 = 1, 0

    main_loop_body = partial(_main_loop_body, verbosity=verbosity)

    while True:
        main_loop_start = time.time()

        idx1, idx2, basis, new_size, xmat_inv = jax.lax.while_loop(
            lambda val: jnp.logical_and(
                jnp.not_equal(val[0], val[3]),
                jnp.not_equal(val[3], basis.shape[0])
            ),
            main_loop_body,
            (idx1, idx2, basis, size, xmat_inv)
        )

        if verbosity > 0:
            print(f'Found {new_size - size} new ops in {time.time() - main_loop_start:.2f}s')

        size = new_size

        if idx1 != size:
            # Need to resize basis and xmat
            if verbosity > 0:
                print(f'Resizing basis array to {max_size + size_increment}')

            max_size += size_increment
            basis = jnp.resize(basis, (max_size,) + basis.shape[1:]).at[size:].set(0.)
            xmat_inv = jnp.linalg.inv(_compute_xmatrix(basis))
            continue

        if max_dim is not None and size >= max_dim:
            basis = basis[:max_dim]

        break

    return [np.asarray(op) for op in basis[:size]]
