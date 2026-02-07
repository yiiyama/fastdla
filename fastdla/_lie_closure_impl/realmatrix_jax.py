# pylint: disable=unused-argument
"""JAX implementation of the Lie closure generator."""
from collections.abc import Callable, Sequence
from functools import partial
from itertools import product
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
    LOG.info('Number of independent generators: %d', basis_size)
    return gens, (basis_is, basis_ra)


def _get_loop_body(
    symm_generators: bool,
    symm_pool: bool,
    print_every: int | None,
    comm_cutoff: float,
    orthonorm_cutoff: float,
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
        idx_op, generators, pool, pool_size = val[:4]
        if symm_generators:
            basis, basis_size = val[4:]
        else:
            basis, basis_size = pool, pool_size

        if print_every > 0:
            jax.lax.cond(
                jnp.equal(idx_op % print_every, 0),
                lambda: jax.debug.callback(callback, idx_op, pool_size),
                lambda: None
            )

        if symm_pool:
            to_matrix = to_s_matrix
        else:
            to_matrix = to_a_matrix
        if symm_generators == symm_pool:
            if symm_generators:
                commutator = partial(commutator_ra, is_ops=True)
            else:
                commutator = commutator_ra
            innerprod = innerprod_ra
        else:
            commutator = commutator_is
            innerprod = innerprod_is

        sharding = jax.typeof(basis).sharding
        if (num_dev := sharding.num_devices) != 0:
            iround = idx_op // num_dev
            duplication = NamedSharding(sharding.mesh, PartitionSpec(None, None, None))
            ops = to_matrix(pool[:, iround])
            comms = commutator(generators[None, ...], ops[:, None])
            comms = comms.reshape((-1,) + comms.shape[2:])
            comms = jax.device_put(comms, duplication)
            incr = num_dev
        else:
            ops = to_matrix(pool[idx_op])
            comms = commutator(generators, ops)
            incr = 1

        comm_norms = jnp.sqrt(innerprod(comms, comms))[..., None]
        comms = jnp.where(comm_norms < comm_cutoff, jnp.zeros_like(comms), comms)

        # pylint: disable-next=unbalanced-tuple-unpacking
        basis, new_size = _gram_schmidt_jnp(comms, basis, basis_size, orthonorm_cutoff, innerprod)

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
        val = (new_idx, generators)
        if symm_generators:
            val += (pool, pool_size, basis, new_size)
        else:
            val += (basis, new_size)
        return val

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

    def inner_loop(loop_body, gens, idx_op, pool, pool_size, basis=None, basis_size=None):
        init = (gens, idx_op, pool, pool_size)
        if basis is not None:
            init += (basis, basis_size)
        max_size = np.prod(init[-2].shape[:-2])

        val = jax.lax.while_loop(
            lambda val: jnp.logical_not(
                (val[1] >= val[3])  # idx_op >= pool_size -> commutator exhausted
                | (val[-1] == max_dim)  # new_size == max_dim -> reached max dim
                | (val[-1] == max_size)  # new_size == array size -> need reallocation
            ),
            loop_body,
            init
        )
        if basis is None:
            return val[1], val[2], val[3]
        return val[1], val[4], val[5]

    # Main loop: generate nested commutators
    bases = list(bases)
    idx_ops = np.zeros(2, dtype=int)
    basis_sizes = np.array([gens.shape[0] for gens in generators])
    labels = ['symmetric', 'antisymmetric']
    while True:  # Outer loop to handle memory reallocation
        main_loop_start = time.time()
        idxs = np.empty((2, 2), dtype=int)
        new_sizes = np.array(basis_sizes)
        # Symmetric generators, symmetric pool -> update antisymmetric
        idxs[0, 0], bases[1], new_sizes[1] = inner_loop(loop_bodies[(True, True)], generators[0],
                                                        idx_ops[0], bases[0], new_sizes[0],
                                                        bases[1], new_sizes[1])
        # Symmetric generators, antisymmetric pool -> update symmetric
        idxs[0, 1], bases[0], new_sizes[0] = inner_loop(loop_bodies[(True, False)], generators[0],
                                                        idx_ops[1], bases[1], new_sizes[1],
                                                        bases[0], new_sizes[0])
        # Antisymmetric generators, symmetric pool -> update symmetric
        idxs[1, 0], bases[0], new_sizes[0] = inner_loop(loop_bodies[(False, True)], generators[1],
                                                        idx_ops[0], bases[0], new_sizes[0])
        # Antisymmetric generators, antisymmetric pool -> update antisymmetric
        idxs[1, 1], bases[1], new_sizes[1] = inner_loop(loop_bodies[(False, False)], generators[1],
                                                        idx_ops[1], bases[1], new_sizes[1])

        LOG.info('Found %d new ops in %.2fs. New size %d',
                 sum(new_sizes) - sum(basis_sizes), time.time() - main_loop_start, sum(new_sizes))

        idx_ops = np.where(np.all(idxs > idx_ops[None, :], axis=0), idxs[:, 0], idx_ops)
        basis_sizes = new_sizes
        if np.all(idx_ops >= basis_sizes) or np.sum(basis_sizes) >= max_dim:
            # Computed all commutators
            break

        # Need to resize basis and xmat
        for ibasis, (basis, basis_size, label) in enumerate(list(zip(bases, basis_sizes, labels))):
            max_size = np.prod(basis.shape[:-2])
            if basis_size < max_size:
                continue
            LOG.debug('Resizing %s basis array to %d', label, max_size + basis_alloc_unit)
            if num_dev != 0:
                local_unit = basis_alloc_unit // num_dev
                bases[ibasis] = jnp.pad(basis, ((0, 0), (0, local_unit), (0, 0), (0, 0)))
            else:
                bases[ibasis] = jnp.pad(basis, ((0, basis_alloc_unit), (0, 0), (0, 0)))

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
    if not isinstance(cutoff, tuple):
        cutoff = (cutoff,) * 2

    (generators_is, generators_ra), (basis_is, basis_ra) = _init_basis(generators, basis_alloc_unit,
                                                                       shard_basis, cutoff[1])

    if generators_is.shape[0] + generators_ra.shape[0] <= 1:
        return np.concatenate([1.j * generators_is, generators_ra.astype(np.complex128)], axis=0)

    # Make the main loop function
    loop_bodies = {flags: _get_loop_body(flags[0], flags[1], print_every, cutoff[0], cutoff[1])
                   for flags in product([True, False], [True, False])}
    # Run the loop
    return _compute_closure((generators_is, generators_ra), (basis_is, basis_ra), max_dim,
                            loop_bodies, basis_alloc_unit)
