"""Fast calculation of dynamical Lie algebra."""

from importlib.metadata import version
__version__ = version(__name__)

from .eigenspace import get_eigenspace
from .lie_closure import lie_closure, orthogonalize
from .pauli import PauliProduct
from .sparse_pauli_sum import SparsePauliSum, SparsePauliSumArray

__all__ = [
    'get_eigenspace',
    'lie_closure', 'orthogonalize',
    'PauliProduct',
    'SparsePauliSum', 'SparsePauliSumArray'
]
