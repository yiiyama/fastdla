# pylint: disable=unused-argument
"""JAX implementation of the Lie closure generator."""
from collections.abc import Callable, Sequence
from functools import partial
import logging
from typing import Optional
import time
import numpy as np
import jax
from jax import Array
import jax.numpy as jnp
from jax.sharding import AxisType, NamedSharding, PartitionSpec
from fastdla.linalg.matrix_ops_jax import (
    innerprod as minnerprod,
    commutator as mcommutator
)
from fastdla.linalg.hermitian_ops_jax import (
    innerprod as hinnerprod,
    commutator as hcommutator,
    to_matrix_stack,
    to_complex_matrix,
    from_complex_matrix
)
from fastdla.algorithms.gram_schmidt import gram_schmidt

LOG = logging.getLogger(__name__)

_gs_update_generic = jax.jit(
    partial(gram_schmidt, innerprod_op=minnerprod, npmod=jnp)
)
_gs_update_skew = jax.jit(
    partial(gram_schmidt, innerprod_op=hinnerprod, npmod=jnp)
)


def _lie_basis(
    ops: Array,
    gs_update: Callable
) -> np.ndarray:
    """Identify a basis for the linear space spanned by ops.

    Args:
        ops: Lie algebra elements whose span to calculate the basis for.

    Returns:
        A list of linearly independent ops.
    """
    ops = jnp.asarray(ops)
    if ops.shape[0] == 0:
        raise ValueError('Cannot determine the basis of null space')

    # Initialize a list of normalized generators
    basis = jnp.zeros_like(ops)
    basis, basis_size = gs_update(ops, basis=basis, basis_size=0)
    return basis[:basis_size]


def lie_basis(
    ops: Sequence[Array],
    *,
    skew_hermitian: bool = False,
    cutoff: float = 1.e-08
) -> np.ndarray:
    if skew_hermitian:
        ops = from_complex_matrix(ops, skew=True)
        gs_update = _gs_update_skew
    else:
        gs_update = _gs_update_generic
    gs_update = partial(gs_update, cutoff=cutoff)

    basis = _lie_basis(ops, gs_update)

    if skew_hermitian:
        basis = to_complex_matrix(basis, skew=True)
    return np.array(basis)


def _init_basis(
    generators: Array,
    gs_update: Callable,
    alloc_unit: int,
    shard: bool
):
    if (num_gen := generators.shape[0]) == 0:
        raise ValueError('Empty set of generators')

    max_size = ((num_gen - 1) // alloc_unit + 1) * alloc_unit

    if shard:
        if alloc_unit % jax.device_count() != 0:
            raise ValueError('Basis allocation unit size is not a multiple of device_count')
        mesh = jax.make_mesh((jax.device_count(),), ('x',), axis_types=(AxisType.Explicit,))
        sharding = NamedSharding(mesh, PartitionSpec('x', None, None, None))
        num_dev = sharding.num_devices
        shard_size = max_size // num_dev
        basis_shape = (num_dev, shard_size) + generators.shape[1:]
    else:
        sharding = jax.devices()[0]
        basis_shape = (max_size,) + generators.shape[1:]

    basis = jnp.zeros(basis_shape, dtype=generators.dtype, device=sharding)
    basis, basis_size = gs_update(generators, basis=basis, basis_size=0)

    if shard:
        # Sharded basis is filled evenly on devices
        # pylint: disable-next=unbalanced-tuple-unpacking
        iround, idev = np.unravel_index(np.arange(basis_size), (shard_size, num_dev))
        duplication = NamedSharding(mesh, PartitionSpec(None, None, None))
        generators = basis.at[idev, iround].get(out_sharding=duplication)
    else:
        generators = basis[:basis_size]
    LOG.info('Number of independent generators: %d', basis_size)
    return generators, basis


def _get_loop_body(
    print_every: int | None,
    gs_update: Callable,
    commutator: Callable
):
    """Make the compiled loop body function using the given algorithm."""
    log_level = LOG.getEffectiveLevel()
    if print_every is None:
        if log_level <= logging.DEBUG:
            print_every = 1
        elif log_level <= logging.INFO:  # 20
            print_every = log_level * 100
        else:
            print_every = -1

    def callback(idx_op, basis_size):
        LOG.log(log_level, 'Calculating commutators with op %d/%d', idx_op, basis_size)

    @jax.jit
    def loop_body(val: tuple) -> tuple:
        """Compute the commutator and update the basis with orthogonal components."""
        idx_op, generators, basis, basis_size = val

        if print_every > 0:
            jax.lax.cond(
                jnp.equal(idx_op % print_every, 0),
                lambda: jax.debug.callback(callback, idx_op, basis_size),
                lambda: None
            )

        sharding = jax.typeof(basis).sharding
        if (num_dev := sharding.num_devices) != 0:
            iround = idx_op // num_dev
            comms = commutator(generators[None, ...], basis[:, iround][:, None])
            comms = comms.reshape((-1,) + basis.shape[2:])
            duplication = NamedSharding(sharding.mesh, PartitionSpec(None, None, None))
            comms = jax.device_put(comms, duplication)
            incr = num_dev
        else:
            comms = commutator(generators, basis[idx_op])
            incr = 1

        basis, new_size = gs_update(comms, basis=basis, basis_size=basis_size)
        # Handle overshoots:
        # Sharded basis
        # Case A: basis array size == new_size
        #   -> Resize the basis array and resume the loop from idx_op
        # Case B: (idx_op + num_dev > old_size > idx_op) and (new_size > old_size)
        #   -> Stay at idx_op
        # Case C: (idx_op + num_dev > old_size > idx_op) and (new_size == old_size)
        #   -> Normal termination
        # Unsharded basis
        # Case A: basis array size == new_size
        #   -> Resize the basis array and resume the loop from idx_op
        max_size = np.prod(basis.shape[:-2])
        idx_op_incr = idx_op + incr
        new_idx = jax.lax.select(
            jnp.equal(new_size, max_size),
            idx_op,
            jax.lax.select(
                jnp.greater(idx_op_incr, basis_size) & jnp.greater(basis_size, idx_op),
                jax.lax.select(
                    jnp.not_equal(new_size, basis_size),
                    idx_op,
                    idx_op_incr
                ),
                idx_op_incr
            )
        )
        return new_idx, generators, basis, new_size

    return loop_body


def _compute_closure(
    generators: Array,
    basis: Array,
    max_dim: int,
    loop_body: Callable,
    basis_alloc_unit: int
):
    num_dev = jax.typeof(basis).sharding.num_devices
    # Main loop: generate nested commutators
    idx_op = 0
    basis_size = generators.shape[0]
    while True:  # Outer loop to handle memory reallocation
        main_loop_start = time.time()
        # Main (inner) loop: iteratively compute the next commutator and update the basis based on
        # the current one
        max_size = np.prod(basis.shape[:-2])
        idx_op, generators, basis, new_size = jax.lax.while_loop(
            lambda val: jnp.logical_not(
                (val[0] >= val[3])  # idx_op >= new_size -> commutator exhausted
                | (val[3] == max_dim)  # new_size == max_dim -> reached max dim
                | (val[3] == max_size)  # new_size == array size -> need reallocation
            ),
            loop_body,
            (idx_op, generators, basis, basis_size)
        )

        LOG.info('Found %d new ops in %.2fs. New size %d',
                 new_size - basis_size, time.time() - main_loop_start, new_size)

        basis_size = new_size
        if idx_op >= basis_size or basis_size == max_dim:
            # Computed all commutators
            break

        # Need to resize basis and xmat
        LOG.debug('Resizing basis array to %d', max_size + basis_alloc_unit)
        if num_dev != 0:
            local_unit = basis_alloc_unit // num_dev
            basis = jnp.pad(basis, ((0, 0), (0, local_unit), (0, 0), (0, 0)))
        else:
            basis = jnp.pad(basis, ((0, basis_alloc_unit), (0, 0), (0, 0)))

    if num_dev != 0:
        # basis[:basis_size] may not fit in one device -> return the sharded array as is along
        # with the indices
        # pylint: disable-next=unbalanced-tuple-unpacking
        indices = np.unravel_index(np.arange(basis_size), basis.shape[:2][::-1])
        return basis, indices[::-1]

    return basis[:basis_size]


def lie_closure(
    generators: Sequence[Array],
    *,
    max_dim: Optional[int] = None,
    print_every: Optional[int] = None,
    skew_hermitian: bool = False,
    cutoff: float = 1.e-08,
    basis_alloc_unit: int = 1024,
    shard_basis: bool = False
) -> Array:
    """Compute the Lie closure of given generators.

    Args:
        generators: Lie algebra elements to compute the closure from.
        max_dim: Cutoff for the dimension of the Lie closure. If set, the algorithm may be halted
            before a full closure is obtained.

    Returns:
        A basis of the Lie algebra spanned by the nested commutators of the generators.
    """
    if skew_hermitian:
        generators = from_complex_matrix(generators, skew=True)
        gs_update = _gs_update_skew
        commutator = partial(hcommutator, skew=True, is_matrix=(True, False))
    else:
        generators = jnp.asarray(generators)
        gs_update = _gs_update_generic
        commutator = mcommutator
    gs_update = partial(gs_update, cutoff=cutoff)

    generators, basis = _init_basis(generators, gs_update, basis_alloc_unit, shard_basis)
    if skew_hermitian:
        generators = to_matrix_stack(generators, skew=True)

    if generators.shape[0] <= 1:
        return generators

    # Make the main loop function
    loop_body = _get_loop_body(print_every, gs_update, commutator)
    # Run the loop
    return _compute_closure(generators, basis, max_dim, loop_body, basis_alloc_unit)
