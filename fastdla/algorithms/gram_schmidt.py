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
from fastdla.linalg.vector_ops import innerprod

LOG = logging.getLogger(__name__)


def orthonormalize(
    vector: NDArray,
    basis: NDArray,
    cutoff: float = 1.e-08,
    innerprod_op: Callable[[NDArray, NDArray], NDArray] = innerprod,
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
    def _orthonormalize(_vector):
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
        projection = tensordot(innerprod_op(_vector, basis).conjugate(), basis, axes)
        orth = _vector - projection
        onorm = npmod.sqrt(innerprod_op(orth, orth))[..., None]
        is_null = npmod.isclose(onorm, 0., atol=cutoff)
        return (npmod.where(is_null, 0., orth) / npmod.where(is_null, 1., onorm),
                npmod.where(is_null[..., 0], 0., onorm[..., 0]))

    orth, vnorm = _orthonormalize(vector)
    if npmod is np:
        LOG.debug('Direct orthogonalization found an orth component with norm %.3e', vnorm)
        if vnorm == 0.:
            return False, np.zeros_like(orth), np.zeros_like(vnorm)

        reorth, renorm = _orthonormalize(orth)
        LOG.debug('Re-orthogonalization found an orth component with norm %.3e', renorm)
    else:
        reorth, renorm = jax.lax.cond(
            jnp.equal(vnorm, 0.),
            lambda _orth, _vnorm: (jnp.zeros_like(_orth), jnp.zeros_like(_vnorm)),
            lambda _orth, _: _orthonormalize(_orth),
            orth, vnorm
        )

    return npmod.isclose(renorm, 1.), reorth, vnorm


def _gram_schmidt_np(
    vectors: NDArray,
    basis: NDArray,
    basis_size: int,
    cutoff: float,
    innerprod_op,
    on_overflow: str = 'raise'
) -> tuple[NDArray, int]:
    for vector in vectors:
        has_orth, orth, _ = orthonormalize(vector, basis, cutoff=cutoff, innerprod_op=innerprod_op)
        if has_orth:
            LOG.debug('Found an orthogonal component. Updating basis to size %d', basis_size + 1)
            if basis_size == basis.shape[0]:
                if on_overflow == 'raise':
                    raise RuntimeError('Basis array overflow')
                if on_overflow == 'warn':
                    warn('Basis array overflow')
            else:
                basis[basis_size] = orth
                basis_size += 1
        else:
            LOG.debug('No orthogonal component found.')

    return basis, basis_size


@partial(jax.jit, static_argnums=[4])
def _gram_schmidt_jnp(
    vectors: NDArray,
    basis: NDArray,
    basis_size: int,
    cutoff: float,
    innerprod_op,
) -> tuple[NDArray, int]:
    sharding = jax.typeof(basis).sharding
    if (num_dev := sharding.num_devices) != 0:
        def update(_vec, _basis, _pos):
            iround = _pos // num_dev
            idev = _pos % num_dev
            return _basis.at[idev, iround].set(_vec, out_sharding=sharding), _pos + 1

    else:
        def update(_vec, _basis, _pos):
            return _basis.at[_pos].set(_vec), _pos + 1

    def loop_body(val):
        ivec, _vectors, _basis, _basis_size = val
        has_orth, orth, _ = orthonormalize(_vectors[ivec], _basis, cutoff=cutoff,
                                           innerprod_op=innerprod_op, npmod=jnp)
        _basis, _basis_size = jax.lax.cond(
            has_orth,
            update,
            lambda v, b, p: (b, p),
            orth, _basis, _basis_size
        )
        return ivec + 1, _vectors, _basis, _basis_size

    max_size = np.prod(basis.shape[:-2])
    basis, basis_size = jax.lax.while_loop(
        lambda val: (val[0] < val[1].shape[0]) & (val[3] < max_size),
        loop_body,
        (0, vectors, basis, basis_size)
    )[2:]

    return basis, basis_size


def gram_schmidt(
    vectors: NDArray,
    basis: Optional[NDArray] = None,
    basis_size: Optional[int] = None,
    cutoff: float = 1.e-08,
    on_overflow: str = 'raise',
    innerprod_op: Callable[[NDArray, NDArray], NDArray] = innerprod,
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
        return _gram_schmidt_np(vectors, basis, basis_size, cutoff, innerprod_op, on_overflow)
    if npmod is jnp:
        return _gram_schmidt_jnp(vectors, basis, basis_size, cutoff, innerprod_op)
