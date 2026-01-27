"""Vector orthogonalization and the Gram-Schmidt process."""
import logging
from collections.abc import Callable
from typing import Optional
import numpy as np
from numpy.typing import NDArray
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None
from fastdla.linalg.vector_ops import innerprod, norm

LOG = logging.getLogger(__name__)


def orthonormalize(
    vector: NDArray,
    basis: NDArray,
    cutoff: float = 1.e-08,
    innerprod_op: Callable[[NDArray, NDArray], NDArray] = innerprod,
    norm_op: Callable[[NDArray], NDArray] = norm,
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
        # What we want is
        #   jnp.tensordot(innerprod(basis, op), basis, [[0], [0]])
        # but we instead compute the conjugate of the innerprod to reduce the number of computation
        projection = npmod.tensordot(innerprod_op(_vector, basis).conjugate(), basis, [[0], [0]])
        orth = _vector - projection
        onorm = norm_op(orth)[..., None]
        is_null = jnp.isclose(onorm, 0., atol=cutoff)
        return (jnp.where(is_null, 0., orth) / jnp.where(is_null, 1., onorm),
                jnp.where(is_null[..., 0], 0., onorm[..., 0]))

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


def _gram_schmidt_update(
    vector: NDArray,
    basis: NDArray,
    basis_size: int | None,
    cutoff: float,
    innerprod_op,
    norm_op,
    npmod=np
) -> tuple[NDArray, int | None]:
    """Identify the orthogonal component and update the basis."""
    has_orth, orth, _ = orthonormalize(vector, basis, cutoff=cutoff, innerprod_op=innerprod_op,
                                       norm_op=norm_op, npmod=npmod)

    if npmod is np:
        if LOG.getEffectiveLevel() <= logging.DEBUG:
            if has_orth:
                LOG.debug('Found an orthogonal component. Updating basis to size %d',
                          basis_size + 1)
            else:
                LOG.debug('No orthogonal component found.')

        if has_orth:
            if basis_size is None:
                # Is only valid for npmod=np
                basis = np.concatenate([basis, orth[None, :]], axis=0)
            else:
                basis[basis_size] = orth
                basis_size += 1
    else:
        basis, basis_size = jax.lax.cond(
            has_orth,
            lambda _vec, _basis, _size: (_basis.at[_size].set(_vec), _size + 1),
            lambda _, _basis, _size: (_basis, _size),
            orth, basis, basis_size
        )

    return basis, basis_size


def gram_schmidt(
    vectors: NDArray,
    basis: Optional[np.ndarray] = None,
    basis_size: Optional[int] = None,
    cutoff: float = 1.e-08,
    innerprod_op: Callable[[NDArray, NDArray], NDArray] = innerprod,
    norm_op: Callable[[NDArray, float], NDArray] = norm,
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
    if npmod is np:
        start = 0
        if basis is None:
            if vectors.shape[0] == 0:
                return vectors.copy()
            basis = norm_op(vectors[0], cutoff=cutoff)[0][None, :]
            basis_size = None
            start = 1

        for vector in vectors[start:]:
            basis, basis_size = _gram_schmidt_update(vector, basis, basis_size, cutoff,
                                                     innerprod_op, norm_op)
    else:
        def update(val):
            ivec, _vectors, _basis, _basis_size = val
            _basis, _basis_size = _gram_schmidt_update(_vectors[ivec], basis, basis_size, cutoff,
                                                       innerprod_op, norm_op, npmod=npmod)
            return ivec + 1, _vectors, _basis, _basis_size

        basis, basis_size = jax.lax.while_loop(
            lambda val: (val[0] < val[1].shape[0]) & (val[3] < val[2].shape[0]),
            update,
            (0, vectors, basis, basis_size)
        )[2:]

    if basis_size is None:
        return basis
    return basis, basis_size
