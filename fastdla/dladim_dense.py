"""Calculation of DLA based on dense matrices."""
from collections.abc import Sequence
from functools import partial
from typing import Optional
import time
import numpy as np
import jax
import jax.numpy as jnp


@jax.jit
def _commutator(op1, op2):
    return op1 @ op2 - op2 @ op1


@jax.jit
def _orthonormalize(new_op, ymat):
    """
    Let the known orthonormal basis ops be P0, P1, ..., Pn. The basis_matrix Π is a matrix formed by
    stacking the column vectors {Pi}:
    Π = (P0, P1, ..., Pn).
    If new_op Q is linearly independent from {Pi}, with Y = Π Π†,
    R = Q - Y Q ≠ 0.
    """
    shape = new_op.shape
    new_op = jnp.ravel(new_op)
    residual = new_op - ymat @ new_op
    norm = jnp.sqrt(jnp.sum(jnp.square(jnp.abs(residual))))
    residual = residual.reshape(shape)
    return jax.lax.cond(
        jnp.isclose(norm, 0.),
        lambda _residual, _: (False, jnp.zeros_like(_residual)),
        lambda _residual, _norm: (True, _residual / _norm),
        residual, norm
    )


@jax.jit
def _continue(_, basis, size, ymat):
    return basis, size, ymat


@jax.jit
def _update(new_op, basis, size, ymat):
    basis = basis.at[size].set(new_op)
    flat = jnp.ravel(new_op)
    ymat += jnp.outer(flat, flat.conjugate())
    return basis, size + 1, ymat


@jax.jit
def _if_independent_update(comm, basis, size, ymat):
    is_independent, new_op = _orthonormalize(comm, ymat)
    return jax.lax.cond(
        is_independent,
        _update,
        _continue,
        new_op, basis, size, ymat
    )


@partial(jax.jit, static_argnames=['verbosity'])
def _main_loop_body(val, verbosity=0):
    idx1, idx2, comm, basis, size, ymat = val

    if verbosity > 1:
        icomm = (idx1 * (idx1 + 1)) // 2 + idx2
        jax.lax.cond(
            jnp.equal(icomm % 2000, 0),
            lambda _size, _icomm, _idx1, _idx2: jax.debug.print(
                'Basis size {size}; {icomm}th commutator [b[{idx1}], b[{idx2}]]',
                size=_size, icomm=_icomm, idx1=_idx1, idx2=_idx2
            ),
            lambda _0, _1, _2, _3: None,
            size, icomm, idx1, idx2
        )

    next_idx1, next_idx2 = jax.lax.cond(
        jnp.equal(idx2 + 1, idx1),
        lambda _idx1, _: (_idx1 + 1, 0),
        lambda _idx1, _idx2: (_idx1, _idx2 + 1),
        idx1, idx2
    )
    # Compute the next commutator in parallel to the independence of the current one
    next_comm = jax.lax.cond(
        jnp.equal(next_idx1, basis.shape[0]),
        lambda op1, _: jnp.empty_like(op1),
        _commutator,
        basis[next_idx1], basis[next_idx2]
    )
    basis, size, ymat = jax.lax.cond(
        jnp.allclose(comm, 0.),
        _continue,
        _if_independent_update,
        comm, basis, size, ymat
    )
    return (next_idx1, next_idx2, next_comm, basis, size, ymat)


def count_dla_dim(
    generators: Sequence[np.ndarray],
    *,
    size_increment: int = 256,
    max_dim: Optional[int] = None,
    verbosity: int = 0
) -> list[np.ndarray]:
    """Only count the DLA dimension.

    When only the dimension is needed, we can work with an orthonormalized basis to simplify the
    linear dependence determination.
    """
    if (size := len(generators)) == 0:
        return []

    max_size = size_increment
    generators = jnp.asarray(generators)
    basis = jnp.zeros((max_size,) + generators.shape[1:], dtype=generators.dtype)
    flat_shape = np.prod(generators.shape[1:])
    ymat = jnp.zeros((flat_shape, flat_shape), dtype=generators.dtype)
    size = 0
    for op in generators:
        basis, size, ymat = _if_independent_update(op, basis, size, ymat)

    idx1, idx2 = 1, 0

    main_loop_body = partial(_main_loop_body, verbosity=verbosity)

    while True:
        main_loop_start = time.time()

        comm = _commutator(basis[idx1], basis[idx2])
        idx1, idx2, _, basis, new_size, ymat = jax.lax.while_loop(
            lambda val: jnp.logical_and(
                jnp.not_equal(val[0], val[4]),
                jnp.not_equal(val[4], basis.shape[0])
            ),
            main_loop_body,
            (idx1, idx2, comm, basis, size, ymat)
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
            continue

        if max_dim is not None and size >= max_dim:
            basis = basis[:max_dim]

        break

    return size
