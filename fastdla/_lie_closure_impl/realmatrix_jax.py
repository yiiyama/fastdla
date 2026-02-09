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
from fastdla.linalg.symmetric_ops_jax import (
    innerprod_is,
    innerprod_ra,
    commutator_is,
    commutator_ra,
    to_s_matrix,
    to_a_matrix,
    from_complex_is_matrix,
    to_complex_is_matrix,
    from_complex_ra_matrix,
    to_complex_ra_matrix
)
from fastdla.algorithms.gram_schmidt import _gram_schmidt_jnp

LOG = logging.getLogger(__name__)


def _lie_basis(
    ops: tuple[Array, Array],
    cutoff
) -> tuple[np.ndarray, np.ndarray]:
    """Identify a basis for the linear space spanned by ops.

    Args:
        ops: Lie algebra elements whose span to calculate the basis for.

    Returns:
        A list of linearly independent ops.
    """
    ops_is, ops_ra = ops
    if ops_is.shape[0] + ops_ra.shape[0] == 0:
        raise ValueError('Cannot determine the basis of null space')

    # Initialize a list of normalized generators
    basis_is = jnp.zeros_like(ops_is)
    basis_is, basis_is_size = _gram_schmidt_jnp(ops_is, basis_is, 0, cutoff, innerprod_is)
    basis_ra = jnp.zeros_like(ops_ra)
    basis_ra, basis_ra_size = _gram_schmidt_jnp(ops_ra, basis_ra, 0, cutoff, innerprod_ra)
    return basis_is[:basis_is_size], basis_ra[:basis_ra_size]


def lie_basis(
    ops: Sequence[Array],
    *,
    cutoff: float = 1.e-08
) -> np.ndarray:
    ops = jnp.asarray(ops)
    is_flags = jnp.all(jnp.isclose(ops.real, 0.), axis=(-2, -1))
    ops_is = from_complex_is_matrix(ops[is_flags])
    ops_ra = from_complex_ra_matrix(ops[~is_flags])
    basis_is, basis_ra = _lie_basis((ops_is, ops_ra), cutoff)
    basis = np.concatenate([to_complex_is_matrix(basis_is), to_complex_ra_matrix(basis_ra)], axis=0)
    return basis


def _init_basis(
    generators: Array,
    alloc_unit: int,
    shard: bool,
    orthonorm_cutoff
):
    if (num_gen := generators.shape[0]) == 0:
        raise ValueError('Empty set of generators')

    is_flags = jnp.all(jnp.isclose(generators.real, 0.), axis=(-2, -1))
    generators_is = from_complex_is_matrix(generators[is_flags])
    generators_ra = from_complex_ra_matrix(generators[~is_flags])

    max_size = ((num_gen - 1) // alloc_unit + 1) * alloc_unit

    if shard:
        if alloc_unit % jax.device_count() != 0:
            raise ValueError('Basis allocation unit size is not a multiple of device_count')
        mesh = jax.make_mesh((jax.device_count(),), ('x',), axis_types=(AxisType.Explicit,))
        sharding = NamedSharding(mesh, PartitionSpec('x', None, None, None))
        num_dev = sharding.num_devices
        shard_size = max_size // num_dev
        basis_is_shape = (num_dev, shard_size) + generators_is.shape[1:]
        basis_ra_shape = (num_dev, shard_size) + generators_ra.shape[1:]
    else:
        sharding = jax.devices()[0]
        basis_is_shape = (max_size,) + generators_is.shape[1:]
        basis_ra_shape = (max_size,) + generators_ra.shape[1:]

    basis_is = jnp.zeros(basis_is_shape, device=sharding)
    basis_is, basis_is_size = _gram_schmidt_jnp(generators_is, basis_is, 0, orthonorm_cutoff,
                                                innerprod_is)
    basis_ra = jnp.zeros(basis_ra_shape, device=sharding)
    basis_ra, basis_ra_size = _gram_schmidt_jnp(generators_ra, basis_ra, 0, orthonorm_cutoff,
                                                innerprod_ra)

    if shard:
        gens = []
        for basis, basis_size, to_matrix_fn in [
            (basis_is, basis_is_size, to_s_matrix), (basis_ra, basis_ra_size, to_a_matrix)
        ]:
            # Sharded basis is filled evenly on devices
            # pylint: disable-next=unbalanced-tuple-unpacking
            iround, idev = np.unravel_index(np.arange(basis_size), (shard_size, num_dev))
            duplication = NamedSharding(mesh, PartitionSpec(None, None, None))
            gens.append(to_matrix_fn(basis.at[idev, iround].get(out_sharding=duplication)))
        gens = tuple(gens)
    else:
        gens = (to_s_matrix(basis_is[:basis_is_size]), to_a_matrix(basis_ra[:basis_ra_size]))
    LOG.info('Number of independent generators: %d symmetric %d antisymmetric',
             basis_is_size, basis_ra_size)
    return gens, (basis_is, basis_ra)


def _get_loop_body(
    ipool: int,
    print_every: int | None,
    comm_cutoff: float,
    orthonorm_cutoff: float,
    batch_size: int = 1
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

    to_matrix = (to_s_matrix, to_a_matrix)[ipool]

    @jax.jit
    def loop_body(val: tuple) -> tuple:
        """Compute the commutator and update the basis with orthogonal components."""
        generators, idx_op, bases, basis_sizes, stop_state = val
        pool = bases[ipool]
        pool_size = basis_sizes[ipool]

        if print_every > 0:
            jax.lax.cond(
                jnp.equal(idx_op % print_every, 0),
                lambda: jax.debug.callback(callback, idx_op, pool_size),
                lambda: None
            )

        sharding = jax.typeof(bases[ipool]).sharding
        num_dev = sharding.num_devices
        duplication = NamedSharding(sharding.mesh, PartitionSpec(None, None, None))

        def compute_and_update(igen):
            if igen == ipool:
                if ipool == 0:
                    commutator = partial(commutator_ra, is_ops=True)
                else:
                    commutator = commutator_ra
                innerprod = innerprod_ra
                basis = bases[1]
                basis_size = basis_sizes[1]
            else:
                commutator = commutator_is
                innerprod = innerprod_is
                basis = bases[0]
                basis_size = basis_sizes[0]

            if num_dev == 0:
                elements = jax.lax.dynamic_slice_in_dim(pool, idx_op, batch_size, axis=0)
                ops = to_matrix(elements)
                comms = commutator(generators[igen][None, ...], ops[:, None])
                comms = comms.reshape((-1,) + comms.shape[2:])
            else:
                iround = idx_op // num_dev
                ops = to_matrix(pool[:, iround])
                comms = commutator(generators[igen][None, ...], ops[:, None])
                comms = comms.reshape((-1,) + comms.shape[2:])
                comms = jax.device_put(comms, duplication)

            comm_norms = jnp.sqrt(innerprod(comms, comms))[..., None]
            comms = jnp.where(comm_norms < comm_cutoff, jnp.zeros_like(comms), comms)
            return _gram_schmidt_jnp(comms, basis, basis_size, orthonorm_cutoff,
                                     innerprod=innerprod)

        bidx = (1 - ipool, ipool)
        new_bases = [None, None]
        new_sizes = [None, None]
        new_bases[bidx[0]], new_sizes[bidx[0]] = compute_and_update(0)
        new_bases[bidx[1]], new_sizes[bidx[1]] = compute_and_update(1)
        bases = tuple(new_bases)

        max_sizes = [np.prod(basis.shape[:-1]) for basis in bases]
        incr = batch_size if num_dev == 0 else num_dev
        # Check if we should increment the idx_op pointer
        # - If either basis array size == new_size
        #   -> Resize the basis array and resume the loop from idx_op (2)
        # - If idx_op + incr >= pool_size (> idx_op)
        #   - If new_size == pool_size
        #     -> Normal termination (1)
        #   - If new_size > pool_size
        #     -> Unchecked ops exist between idx_op and idx_op + incr. Do not increment (0)
        # - If pool_size > idx_op + incr
        #   -> Normal increment (0)
        idx_op, stop_state = jax.lax.cond(
            jnp.logical_or(
                jnp.equal(new_sizes[0], max_sizes[0]),
                jnp.equal(new_sizes[1], max_sizes[1])
            ),
            lambda: (idx_op, 2),
            lambda: (
                jax.lax.cond(
                    jnp.greater_equal(idx_op + incr, pool_size),
                    lambda: (
                        jax.lax.cond(
                            jnp.equal(new_sizes[ipool], pool_size),
                            lambda: (pool_size, 1),
                            lambda: (idx_op, 0)
                        )
                    ),
                    lambda: (idx_op + incr, 0)
                )
            )
        )
        return generators, idx_op, tuple(bases), tuple(new_sizes), stop_state

    return loop_body


def _compute_closure(
    generators: tuple[Array, Array],
    bases: tuple[Array, Array],
    max_dim: int,
    loop_bodies: dict[tuple[bool, bool], Callable],
    basis_alloc_unit: int
):
    sharding = jax.typeof(bases[0]).sharding
    num_dev = sharding.num_devices
    labels = ['symmetric', 'antisymmetric']

    def run_loop(loop_body, idx_op, bases, basis_sizes):
        init = (generators, idx_op, bases, basis_sizes, 0)
        _, idx_op, bases, basis_sizes, stop_state = jax.lax.while_loop(
            lambda val: jnp.equal(val[4], 0),
            loop_body,
            init
        )
        print(idx_op, [b.shape[0] for b in bases], basis_sizes, stop_state, stop_state == 2)
        if stop_state == 2:
            # Need to resize basis
            bases = list(bases)
            for ib, (basis, basis_size, label) in enumerate(list(zip(bases, basis_sizes, labels))):
                if basis_size < (max_size := np.prod(basis.shape[:-1])):
                    continue
                LOG.debug('Resizing %s basis array to %d', label, max_size + basis_alloc_unit)
                if num_dev != 0:
                    local_unit = basis_alloc_unit // num_dev
                    bases[ib] = jnp.pad(basis, ((0, 0), (0, local_unit), (0, 0)))
                else:
                    bases[ib] = jnp.pad(basis, ((0, basis_alloc_unit), (0, 0)))
            bases = tuple(bases)

        return idx_op, bases, basis_sizes

    # Main loop: generate nested commutators
    idx_ops = np.zeros(2, dtype=int)
    basis_sizes = np.array([gens.shape[0] for gens in generators])
    while True:
        for ib, (idx, loop_body, label) in enumerate(zip(idx_ops, loop_bodies, labels)):
            if idx_ops[ib] == basis_sizes[ib]:
                continue
            loop_start = time.time()
            idx_ops[ib], bases, new_sizes = run_loop(loop_body, idx, bases, tuple(basis_sizes))
            LOG.info('Found %d new ops from %s pool in %.2fs. New total size %d',
                     sum(new_sizes) - sum(basis_sizes), label, time.time() - loop_start,
                     sum(new_sizes))
            basis_sizes = np.array(new_sizes)

        if np.all(idx_ops == basis_sizes) or (max_dim and np.sum(basis_sizes) >= max_dim):
            break

    if num_dev != 0:
        # basis[:basis_size] may not fit in one device -> return the sharded array as is along
        # with the indices
        # pylint: disable-next=unbalanced-tuple-unpacking
        result = []
        for basis, basis_size in zip(bases, basis_sizes):
            indices = np.unravel_index(np.arange(basis_size), basis.shape[:2][::-1])
            result += [basis, indices[::-1]]
        return tuple(result)

    return bases[0][:basis_sizes[0]], bases[1][:basis_sizes[1]]


def lie_closure(
    generators: Sequence[Array],
    *,
    max_dim: Optional[int] = None,
    print_every: Optional[int] = None,
    cutoff: float | tuple[float, float] = 1.e-08,
    basis_alloc_unit: int = 1024,
    shard_basis: bool = False,
    batch_size: int = 1
) -> Array:
    """Compute the Lie closure of given generators.

    Args:
        generators: Lie algebra elements to compute the closure from.
        max_dim: Cutoff for the dimension of the Lie closure. If set, the algorithm may be halted
            before a full closure is obtained.

    Returns:
        A basis of the Lie algebra spanned by the nested commutators of the generators.
    """
    if not isinstance(cutoff, tuple):
        cutoff = (cutoff,) * 2

    generators, bases = _init_basis(generators, basis_alloc_unit, shard_basis, cutoff[1])

    if generators[0].shape[0] + generators[1].shape[0] <= 1:
        return np.concatenate([1.j * generators[0], generators[1].astype(np.complex128)], axis=0)

    # Make the main loop function
    loop_bodies = [_get_loop_body(ipool, print_every, cutoff[0], cutoff[1], batch_size)
                   for ipool in [0, 1]]
    # Run the loop
    return _compute_closure(generators, bases, max_dim, loop_bodies, basis_alloc_unit)
