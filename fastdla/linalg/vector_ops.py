"""Numpy-based ops for stacks of 1D vectors."""
import numpy as np
from numpy.typing import NDArray


def innerprod(vec1: NDArray, vec2: NDArray, npmod=np) -> NDArray:
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


def norm(vector: NDArray, npmod=np) -> NDArray:
    return npmod.sqrt(npmod.sum(npmod.square(npmod.abs(vector)), axis=-1))


def normalize(
    vector: NDArray,
    cutoff: float = 1.e-08
) -> tuple[NDArray, float]:
    """Normalize a vector.

    Args:
        vector: Vector to normalize.
        cutoff: Cutoff for norm of the orthogonal component.
        norm_op: A function that computes the inner product of two vectors.

    Returns:
        Normalized vector and the norm of the original vector.
    """
    vector_norm = norm(vector)[..., None]
    is_null = np.isclose(vector_norm, 0., atol=cutoff)
    return (np.where(is_null, 0., vector) / np.where(is_null, 1., vector_norm),
            np.where(is_null, 0., vector_norm[..., 0]))


def project(
    vector: NDArray,
    basis: NDArray
) -> NDArray:
    """Compute the projection of a vector onto a subspace.

    Args:
        vector: Vector to orthogonalize.
        basis: An array of orthonormal vectors. The first dimension is the size of the basis.

    Returns:
        The vector projected onto the subspace spanned by the basis.
    """
    return np.tensordot(innerprod(basis, vector), basis, [[0], [0]])
