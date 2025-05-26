"""Symmetry operations common to 1D spin-chain models."""
from typing import Optional
import numpy as np
from ..eigenspace import LinearOpFunction, get_eigenspace


def magnetization_projection(
    magnetization: int,
    num_spins: int,
    npmod=np
) -> LinearOpFunction:
    """Return a function that projects out the eigensubspace of magnetization."""
    indices = npmod.arange(2 ** num_spins)
    bin_indices = (indices[:, None] >> npmod.arange(num_spins)[None, ::-1]) % 2
    eigvals = npmod.sum(1 - 2 * bin_indices, axis=1)
    target_states = npmod.nonzero(eigvals == magnetization)[0]

    def op(basis):
        if npmod is np:
            projected = np.array(basis)
            projected[target_states] = 0.
        else:
            basis = npmod.asarray(basis)
            projected = basis.at[target_states].set(0.)
        return projected

    return op


def magnetization_eigenspace(
    magnetization: int,
    basis: Optional[np.ndarray] = None,
    num_spins: Optional[int] = None,
    npmod=np
) -> np.ndarray:
    if num_spins is None:
        num_spins = np.round(np.log2(basis.shape[0])).astype(int)
    op = magnetization_projection(magnetization, num_spins, npmod=npmod)
    return get_eigenspace(op, basis, dim=2 ** num_spins, npmod=npmod)


def parity_reflection(
    num_spins: int
) -> tuple[LinearOpFunction, int]:
    """Return a function that applies a parity reflection to states."""
    if num_spins % 2:
        raise ValueError('Parity undefined for odd number of spins')

    def op(basis):
        basis = basis.reshape((2,) * num_spins + (-1,))
        basis = basis.transpose(tuple(range(num_spins - 1, -1, -1)) + (num_spins,))
        basis = basis.reshape((2 ** num_spins, -1))
        return basis

    return op


def parity_eigenspace(
    parity: int,
    basis: Optional[np.ndarray] = None,
    num_spins: Optional[int] = None,
    npmod=np
) -> np.ndarray:
    """Extract the parity eigenspace."""
    if abs(parity) != 1:
        raise ValueError('Invalid parity eigenvalue')

    if num_spins is None:
        num_spins = np.round(np.log2(basis.shape[0])).astype(int)

    op = parity_reflection(num_spins)
    return get_eigenspace((op, parity), basis, dim=2 ** num_spins, npmod=npmod)
