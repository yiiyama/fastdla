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
