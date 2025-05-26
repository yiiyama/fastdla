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


def translation(
    jphase: int,
    num_spins: int,
    shift: int = 1,
    npmod=np
) -> tuple[LinearOpFunction, complex]:
    r"""Return a function that applies the translation :math:`T_s` to state vectors.

    Args:
        jphase: Integer :math:`j` of the :math:`T_s` eigenvalue :math:`e^{2\pi i j s / N}`.
        num_spins: Number of spins :math:`N`.
        shift: Unit of translation :math:`s`.

    Returns:
        A function that takes a basis matrix :math:`B` and computes :math:`T_s B` and the eigenvalue
        :math:`e^{2\pi i j s / N}`.
    """
    if num_spins % shift:
        raise ValueError('num_spins must be a multiple of shift')

    num_steps = num_spins // shift

    if jphase not in list(range(num_steps)):
        raise ValueError('Invalid jphase value')

    def op(basis):
        """Translate the states in the basis."""
        translated = npmod.array(basis).reshape((2,) * num_spins + (-1,))
        src = np.arange(num_spins)
        dest = np.roll(np.arange(num_spins), -shift)
        translated = npmod.moveaxis(translated, src, dest)
        translated = translated.reshape((2 ** num_spins, -1))
        return translated

    eigval = np.exp(2.j * np.pi / num_steps * jphase)
    return op, eigval


def translation_eigenspace(
    jphase: int,
    basis: Optional[np.ndarray] = None,
    num_spins: Optional[int] = None,
    shift: int = 1,
    npmod=np
) -> np.ndarray:
    r"""Extract the :math:`j`th eigenspace of the translation :math:`T_s`.

    Args:
        jphase: Integer :math:`j` of the :math:`T_s` eigenvalue :math:`e^{2\pi i j s /N}`.
        basis: The basis matrix :math:`B`.
        num_spins: Number of spins :math:`N`.
        shift: Unit of translation :math:`s`.

    Returns:
        A matrix whose columns form the orthonormal basis of the eigen-subspace.
    """
    if num_spins is None:
        num_spins = np.round(np.log2(basis.shape[0])).astype(int)
    op, eigval = translation(jphase, num_spins, shift=shift, npmod=npmod)
    return get_eigenspace((op, eigval), basis, dim=2 ** num_spins, npmod=npmod)
