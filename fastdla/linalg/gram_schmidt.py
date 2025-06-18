"""Vector orthogonalization and the Gram-Schmidt process."""
from collections.abc import Callable
from typing import Optional
import numpy as np
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None


def inner_product(vec1: np.ndarray, vec2: np.ndarray, npmod=np) -> np.ndarray:
    r"""Inner product between two (stacked) vectors.

    Calculates :math:`v_1^{\dagger} v_2`.

    Args:
        vec1: Left-hand side vector :math:`v_1`.
        vec2: Right-hand side vector :math:`v_2`.

    Returns:
        The inner product. If the arguments have extra dimensions, the returned array will have a
        shape of an outer product of these dimensions.
    """
    return npmod.tensordot(vec1.conjugate(), vec2, [[-1], [-1]])


def orthogonalize(
    vector: np.ndarray,
    basis: np.ndarray,
    innerprod: Callable[[np.ndarray, np.ndarray], complex] = inner_product,
    npmod=np
) -> np.ndarray:
    """Compute the orthogonal component of a vector with respect to an orthonormal basis.

    Args:
        vector: Vector to orthogonalize.
        basis: An array of orthonormal vectors. The first dimension is the size of the basis.
        innerprod: A function that computes the inner product of two vectors.

    Returns:
        The orthogonal component of the vector.
    """
    return vector - npmod.tensordot(innerprod(basis, vector), basis, [[0], [0]])


def normalize(
    vector: np.ndarray,
    innerprod: Callable[[np.ndarray, np.ndarray], complex] = inner_product,
    npmod=np
) -> tuple[np.ndarray, float]:
    """Normalize a vector.

    Args:
        vector: Vector to normalize.
        innerprod: A function that computes the inner product of two vectors.

    Returns:
        Normalized vector and the norm of the original vector.
    """
    norm = npmod.sqrt(innerprod(vector, vector))
    is_null = npmod.isclose(norm, 0.)
    return npmod.where(is_null, 0., vector) / npmod.where(is_null, 1., norm), norm


def orthonormalize(
    vector: np.ndarray,
    basis: np.ndarray,
    innerprod: Callable[[np.ndarray, np.ndarray], complex] = inner_product,
    npmod=np
) -> tuple[bool, np.ndarray, float]:
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
    orth = orthogonalize(vector, basis, innerprod=innerprod, npmod=npmod)
    orth, norm = normalize(orth, innerprod=innerprod, npmod=npmod)
    reorth = orthogonalize(orth, basis, innerprod=innerprod, npmod=npmod)
    reorth, renorm = normalize(reorth, innerprod=innerprod, npmod=npmod)
    return npmod.isclose(renorm, 1.), reorth, norm


def _gram_schmidt_update(
    vector: np.ndarray,
    basis: np.ndarray,
    basis_size: Optional[int] = None,
    innerprod: Callable[[np.ndarray, np.ndarray], complex] = inner_product,
    npmod=np
) -> tuple[np.ndarray, int | None]:

    has_orth, orth, _ = orthonormalize(vector, basis, innerprod=innerprod, npmod=npmod)

    if npmod is np:
        if has_orth:
            if basis_size is None:
                # Is only valid for npmod=np
                basis = np.concatenate([basis, orth[None, :]], axis=0)
            else:
                basis[basis_size] = orth
                basis_size += 1
    elif npmod is jnp:
        basis, basis_size = jax.lax.cond(
            has_orth,
            lambda _vec, _basis, _size: (_basis.at[_size].set(_vec), _size + 1),
            lambda _, _basis, _size: (_basis, _size),
            orth, basis, basis_size
        )

    return basis, basis_size


def gram_schmidt(
    vectors: np.ndarray,
    basis: Optional[np.ndarray] = None,
    basis_size: Optional[int] = None,
    innerprod: Callable[[np.ndarray, np.ndarray], complex] = inner_product,
    npmod=np
) -> np.ndarray:
    """Construct an orthonormal basis from an array of vectors through the Gram-Schmidt process.

    If the optional basis is given, the function completes this basis with the given vectors.

    Args:
        vectors: Array of vectors.
        basis: Partially completed basis.
        basis_size: Effective only when `basis` is not None. Current size of the partially completed
            basis. If given as an int, newly found basis vectors are placed into the basis array
            starting from this position. If None, the new vectors are concatenated to the basis
            array.
        innerprod: A function that computes the inner product of two vectors.

    Returns:
        A full array of orthonormal vectors that span the space that is spanned by the given vectors
        and basis.
    """
    start = 0
    if basis is None:
        if vectors.shape[0] == 0:
            return npmod.empty((0, vectors.shape[1]), dtype=vectors.dtype)
        basis = normalize(vectors[0], innerprod=innerprod, npmod=npmod)[0][None, :]
        basis_size = None
        start = 1

    for vector in vectors[start:]:
        basis, basis_size = _gram_schmidt_update(vector, basis, basis_size=basis_size,
                                                 innerprod=innerprod, npmod=npmod)

    return basis
