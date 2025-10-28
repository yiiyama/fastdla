# pylint: disable=import-outside-toplevel
"""Generator of Lie closure."""
from collections.abc import Sequence
import logging
from typing import Any, Optional
from fastdla.sparse_pauli_sum import SparsePauliSum, SparsePauliSumArray

LOG = logging.getLogger(__name__)
AlgebraElement = Any
Basis = Sequence[AlgebraElement]


def lie_basis(
    ops: Basis,
    *,
    keep_original: bool = False,
    **kwargs
) -> tuple[Basis, Basis] | Basis:
    r"""Compute a basis of the linear space spanned by ops.

    Args:
        generators: Lie algebra elements whose span to calculate the basis for.
        keep_original: Whether the returned array of Lie algebra elements should contain the
            original (normalized) generators. If False, the orthonormalized basis used internally in
            the algorithm is returned.

    Returns:
        A list of linearly independent ops and an orthonormal basis if keep_original=True,
        otherwise only the orthonormal basis.
    """
    if isinstance(ops, list) and isinstance(ops[0], SparsePauliSum):
        ops = SparsePauliSumArray(ops)

    if isinstance(ops, SparsePauliSumArray):
        from fastdla._lie_closure_impl.sparse_numba import lie_basis as fn
    else:
        from fastdla._lie_closure_impl.matrix_jax import lie_basis as fn

    return fn(ops, keep_original=keep_original, **kwargs)


def lie_closure(
    generators: Any,
    *,
    keep_original: bool = False,
    max_dim: Optional[int] = None,
    **kwargs
) -> tuple[Basis, Basis] | Basis:
    """Compute the Lie closure of given generators.

    Lie closure generation follows the orthonormalization algorithm in *arXiv:2506.01120*:

    .. code-block:: python

        V0 = []
        for g in G:
            g_perp = orthogonalize(g, V0)
            if g_perp != 0:
                V0.append(g_perp / norm(g_perp))

        V = V0.copy()

        Vprev = V0
        Vnew = []
        while True:
            for g, h in product(V0, Vprev):
                i = commutator(g, h)
                i_perp = orthogonalize(i, V)
                if i_perp != 0:
                    Vnew.append(i_perp / norm(i_perp))

            if len(Vnew) == 0:
                break
            V += Vnew
            Vprev = Vnew
            Vnew = []

    If keep_original is True, there will be an additional list ``B`` which stores the normalized
    nested commutators ``h``. The function then returns both ``B`` and ``V``.

    The inputs to this function can be given in the matrix or SparsePauliSum representations. If
    matrices are given, JAX-based implementation will be called.

    Args:
        generators: Lie algebra elements to compute the closure from.
        keep_original: Whether the returned array of Lie algebra elements should contain the
            original (normalized) generators and their actual nested commutators. If False, the
            orthonormalized basis used internally in the algorithm is returned.
        max_dim: Cutoff for the dimension of the Lie closure. If set, the algorithm may be halted
            before a full closure is obtained.

    Returns:
        A list of linearly independent nested commutators and the orthonormal basis if
        keep_original=True, otherwise only the orthonormal basis.
    """
    if isinstance(generators, list) and isinstance(generators[0], SparsePauliSum):
        generators = SparsePauliSumArray(generators)

    if isinstance(generators, SparsePauliSumArray):
        from fastdla._lie_closure_impl.sparse_numba import lie_closure as fn
    else:
        from fastdla._lie_closure_impl.matrix_jax import lie_closure as fn

    return fn(generators, keep_original=keep_original, max_dim=max_dim, **kwargs)
