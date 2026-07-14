"""Vector orthogonalization and the Gram-Schmidt process."""
import logging
from collections.abc import Callable
from functools import partial
from warnings import warn
from typing import Optional
import numpy as np
from numpy.typing import NDArray
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

LOG = logging.getLogger(__name__)


def orthonormalize(
    vector: NDArray,
    basis: NDArray,
    cutoff: float = 1.e-08,
    innerprod: Optional[Callable[[NDArray, NDArray], NDArray]] = None,
    npmod=np
) -> tuple[bool, NDArray, float]:
    """Normalize the orthogonal component of a vector with respect to a basis.

    To ensure that the normalized vector is actually the orthogonal component and not an artifact
    of floating-point precision, we re-orthogonalize the initial orthonormal vector. Returned flag
    is True only if the norm of the re-orthogonalized vector is close to 1.

    Args:
        vector: Vector to orthonormalize against the basis.
        basis: An array of orthonormal vectors. The first dimension is the size of the basis.
        innerprod: A function that computes the inner product of two vectors.

    Returns:
        A flag indicating the existence of an orthogonal component, the orthonormalized vector, and
        the norm of the orthogonal component.
    """
    if innerprod is None:
        innerprod = npmod.vecdot

    def _orth_and_norm2(_vector):
        if npmod is jnp and (sharding := jax.typeof(basis).sharding).num_devices != 0:
            sharding = NamedSharding(sharding.mesh, PartitionSpec(*((None,) * _vector.ndim)))
            tensordot = partial(jnp.tensordot, out_sharding=sharding)
            axes = [[0, 1], [0, 1]]
        else:
            tensordot = npmod.tensordot
            axes = [[0], [0]]
        # What we want is
        #   tensordot(innerprod(basis, op), basis, [[0], [0]])
        # but we instead compute the conjugate of the innerprod to reduce the number of computation
        projection = tensordot(innerprod(_vector, basis).conjugate(), basis, axes)
        orth = _vector - projection
        onorm2 = innerprod(orth, orth)
        return orth, onorm2

    orth, onorm2 = _orth_and_norm2(vector)
    is_null = npmod.isclose(onorm2, 0., atol=cutoff**2)
    if npmod is np:
        LOG.debug('Direct orthogonalization found an orth component with norm^2 %.3e', onorm2)
        if is_null:
            return False, np.zeros_like(orth), 0.

        reorth, renorm2 = _orth_and_norm2(orth)
        LOG.debug('Re-orthogonalization found an orth component with norm^2 %.3e', renorm2)
        if not np.isclose(renorm2, onorm2):
            return False, np.zeros_like(reorth), 0.

        return True, reorth / np.sqrt(renorm2), onorm2
    elif npmod is jnp:
        def reorthogonalize(_orth):
            reorth, renorm2 = _orth_and_norm2(_orth)
            return jax.lax.cond(
                jnp.isclose(renorm2, onorm2),
                lambda ro, rn2, n2: (True, ro / jnp.sqrt(rn2), n2),
                lambda ro, rn2, n2: (False, jnp.zeros_like(ro), n2),
                reorth, renorm2, onorm2
            )

        return jax.lax.cond(
            is_null,
            lambda _orth: (False, jnp.zeros_like(_orth), 0.),
            reorthogonalize,
            orth
        )


def orthonormalize2(
    vector: NDArray,
    basis: NDArray,
    basis_size: NDArray,
    cutoff: float = 1.e-08,
    innerprod: Optional[Callable[[NDArray, NDArray], NDArray]] = None,
    npmod=np
) -> tuple[bool, NDArray, float]:
    """Modified Gram-Schmidt orthogonalization."""
    if innerprod is None:
        innerprod = npmod.vecdot

    def remove_component(ib, val):
        vec, basis = val
        vec -= innerprod(basis[ib], vec) * basis[ib]
        return vec, basis

    if npmod is np:
        raise NotImplementedError('Under construction')
    else:
        orth, _ = jax.lax.fori_loop(
            0, basis_size,
            remove_component,
            (vector, basis)
        )

    onorm2 = innerprod(orth, orth)
    is_null = npmod.isclose(onorm2, 0., atol=cutoff**2)
    result = npmod.where(is_null, 0., orth) / npmod.where(is_null, 1., npmod.sqrt(onorm2))
    return npmod.logical_not(is_null), result, onorm2


def _gram_schmidt_np(
    vectors: NDArray,
    basis: NDArray,
    basis_size: int,
    cutoff: float,
    innerprod: Optional[Callable] = None,
    on_overflow: str = 'raise'
) -> tuple[NDArray, int]:
    for vector in vectors:
        has_orth, orth, _ = orthonormalize(vector, basis, cutoff=cutoff, innerprod=innerprod)
        if has_orth:
            LOG.debug('Found an orthogonal component. Updating basis to size %d', basis_size + 1)
            if basis_size == basis.shape[0]:
                match on_overflow:
                    case 'raise':
                        raise RuntimeError('Basis array overflow')
                    case 'warn':
                        warn('Basis array overflow')
                    case 'extend':
                        basis = np.concatenate([basis, orth[None, :]], axis=0)
                        basis_size += 1
            else:
                basis[basis_size] = orth
                basis_size += 1
        else:
            LOG.debug('No orthogonal component found.')

    return basis, basis_size


@partial(jax.jit, static_argnames=['innerprod', 'monitor_onorms'])
def _gram_schmidt_jnp(
    vectors: NDArray,
    basis: NDArray,
    basis_size: int,
    cutoff: float,
    innerprod: Optional[Callable] = None,
    monitor_onorms: bool = False
) -> tuple[NDArray, int]:
    if vectors.shape[0] == 0:
        return basis, basis_size

    sharding = jax.typeof(basis).sharding
    if (num_dev := sharding.num_devices) != 0:
        max_size = np.prod(basis.shape[:2])

        def update(_vec, _basis, _pos):
            iround = _pos // num_dev
            idev = _pos % num_dev
            return _basis.at[idev, iround].set(_vec, out_sharding=sharding), _pos + 1

    else:
        max_size = basis.shape[0]

        def update(_vec, _basis, _pos):
            return _basis.at[_pos].set(_vec), _pos + 1

    def loop_body(val):
        ivec, _vectors, _basis, _basis_size = val[:4]
        if monitor_onorms:
            onorms, oflags = val[4:]
        has_orth, orth, onorm2 = orthonormalize(_vectors[ivec], _basis, cutoff=cutoff,
                                                innerprod=innerprod, npmod=jnp)
        # has_orth, orth, onorm2 = orthonormalize2(_vectors[ivec], _basis, _basis_size, cutoff=cutoff,
        #                                          innerprod=innerprod, npmod=jnp)
        _basis, _basis_size = jax.lax.cond(
            has_orth,
            update,
            lambda v, b, p: (b, p),
            orth, _basis, _basis_size
        )
        val = (ivec + 1, _vectors, _basis, _basis_size)
        if monitor_onorms:
            val += (onorms.at[ivec].set(jnp.sqrt(onorm2)), oflags.at[ivec].set(has_orth))
        return val

    init = (0, vectors, basis, basis_size)
    if monitor_onorms:
        init += (jnp.empty(vectors.shape[0]), jnp.empty(vectors.shape[0], dtype=np.bool))

    result = jax.lax.while_loop(
        lambda val: (val[0] < val[1].shape[0]) & (val[3] < max_size),
        loop_body,
        init
    )
    return result[2:]


def gram_schmidt(
    vectors: NDArray,
    basis: Optional[NDArray] = None,
    basis_size: Optional[int] = None,
    cutoff: float = 1.e-08,
    on_overflow: str = 'raise',
    innerprod: Optional[Callable[[NDArray, NDArray], NDArray]] = None,
    npmod=np
) -> NDArray | tuple[NDArray, int]:
    """Construct an orthonormal basis from an array of vectors through the Gram-Schmidt process.

    If the optional basis is given, the function completes this basis with the given vectors.

    Args:
        vectors: Array of vectors.
        basis: Partially completed basis.
        basis_size: Effective only when `basis` is not None. Current size of the partially completed
            basis. If given as an int, newly found basis vectors are placed into the basis array
            starting from this position. If None, the new vectors are concatenated to the basis
            array.

    Returns:
        A full array of orthonormal vectors that span the space that is spanned by the given vectors
        and basis.
    """
    if basis is None:
        basis = npmod.zeros_like(vectors)
        basis_size = 0

    if npmod is np:
        if basis_size is None:
            basis_size = basis.shape[0]

        if innerprod is None:
            match vectors.ndim:
                case 2:
                    innerprod = np.vecdot
                case 3:
                    # pylint: disable-next=function-redefined
                    def innerprod(op1, op2):
                        return np.vecdot(
                                op1.reshape(op1.shape[:-2] + (-1,)),
                                op2.reshape(op2.shape[:-2] + (-1,))
                            ) / op1.shape[-1]
                case _:
                    raise NotImplementedError(f'Innerprod for {vectors.ndim}-dim arrays')

        return _gram_schmidt_np(vectors, basis, basis_size, cutoff, innerprod=innerprod,
                                on_overflow=on_overflow)
    if npmod is jnp:
        if basis_size is None:
            raise ValueError('basis_size is required')

        if innerprod is None:
            match vectors.ndim:
                case 2:
                    innerprod = jnp.vecdot
                case 3:
                    # pylint: disable-next=import-outside-toplevel
                    from fastdla.linalg.matrix_ops_jax import innerprod as _innerprod
                    innerprod = _innerprod
                case _:
                    raise NotImplementedError(f'Innerprod for {vectors.ndim}-dim arrays')

        return _gram_schmidt_jnp(vectors, basis, basis_size, cutoff, innerprod=innerprod)
