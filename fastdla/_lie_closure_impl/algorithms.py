"""Algorithm for linear independence check and basis update."""
from enum import IntEnum


class Algorithms(IntEnum):
    """Algorithm for linear independence check and basis update."""

    GS_DIRECT = 1
    """
    Linear independence check via Gram Schmidt orthogonalization, with basis constructed directly
    from the orthonormalized ops.
    """
    GRAM_SCHMIDT = 2
    """
    Linear independence check via Gram Schmidt orthogonalization, with a separate list of
    independent ops as the basis.
    """
    MATRIX_INV = 3
    """
    Linear independence check via the matrix inversion method.
    """
    SVD = 4
    """
    Linear independence check via SVD of the trial basis matrix.
    """
