"""Fast calculation of dynamical Lie algebra."""

from importlib.metadata import version
__version__ = version(__name__)

from .eigenspace import get_eigenspace
from .lie_closure import linear_independence, lie_closure
from .sparse_pauli_vector import SparsePauliVector, SparsePauliVectorArray

__all__ = [
    'get_eigenspace',
    'linear_independence', 'lie_closure',
    'SparsePauliVector', 'SparsePauliVectorArray'
]
