"""Symmetry operations common to 1D spin-chain models."""
from typing import Optional
import numpy as np
try:
    import jax.numpy as jnp
except ImportError:
    jnp = None
else:
    import jax
from fastdla.algorithms.eigenspace import LinearOpFunction, get_eigenspace


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
            projected = basis.at[target_states].set(0.)
        return projected

    if npmod is jnp:
        op = jax.jit(op)

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
    num_spins: int,
    reflect_about: Optional[int | tuple[int, int]] = None,
    npmod=np
) -> tuple[LinearOpFunction, int]:
    """Return a function that applies a parity reflection to states.

    Parity reflection is uniquely defined under open boundary conditions, but there is a freedom
    to choose the reflection point for periodic boundary conditions. Spins are indexed from right to
    left.

    Args:
        num_spins: Number of spins.
        reflect_about: If an integer n, perform the parity reflection that fixes the nth spin. If
            a tuple of contiguous integers, perform the reflection about the link between the two
            spins.
    """
    if reflect_about is None:
        if num_spins % 2 == 0:
            reflect_about = (num_spins // 2 - 1, num_spins // 2)
        else:
            reflect_about = num_spins // 2

    if isinstance(reflect_about, int):
        # Reverse the order and shift by N-1-2r
        shift = num_spins - 1 - 2 * reflect_about
        dest = np.roll(np.arange(num_spins)[::-1], shift)
    else:
        if num_spins % 2 == 1:
            raise ValueError('Parity reflection about a link undefined for odd number of spins')
        dest = np.roll(np.arange(num_spins)[::-1], num_spins - 2 * max(reflect_about))

    def op(basis):
        basis = npmod.reshape(basis, (2,) * num_spins + (-1,), copy=True)
        basis = npmod.moveaxis(basis, np.arange(num_spins), dest)
        basis = basis.reshape((2 ** num_spins, -1))
        return basis

    if npmod is jnp:
        op = jax.jit(op)

    return op


def parity_eigenspace(
    sign: int,
    basis: Optional[np.ndarray] = None,
    num_spins: Optional[int] = None,
    reflect_about: Optional[int | tuple[int, int]] = None,
    npmod=np
) -> np.ndarray:
    """Extract the parity eigenspace."""
    if abs(sign) != 1:
        raise ValueError('Invalid parity eigenvalue')

    if num_spins is None:
        num_spins = np.round(np.log2(basis.shape[0])).astype(int)

    op = parity_reflection(num_spins, reflect_about=reflect_about, npmod=npmod)
    return get_eigenspace((op, sign), basis, dim=2 ** num_spins, npmod=npmod)


def translation(
    num_spins: int,
    shift: int = 1,
    npmod=np
) -> LinearOpFunction:
    r"""Return a function that applies the translation :math:`T_s` to state vectors.

    Args:
        num_spins: Number of spins :math:`N`.
        shift: Unit of translation :math:`s`.

    Returns:
        A function that takes a basis matrix :math:`B` and computes :math:`T_s B`.
    """
    if num_spins % shift:
        raise ValueError('num_spins must be a multiple of shift')

    def op(basis):
        """Translate the states in the basis."""
        basis = npmod.reshape(basis, (2,) * num_spins + (-1,), copy=True)
        src = np.arange(num_spins)
        dest = np.roll(np.arange(num_spins), -shift)
        basis = npmod.moveaxis(basis, src, dest)
        basis = basis.reshape((2 ** num_spins, -1))
        return basis

    if npmod is jnp:
        op = jax.jit(op)

    return op


def translation_eigenspace(
    jphase: int,
    basis: Optional[np.ndarray] = None,
    num_spins: Optional[int] = None,
    shift: int = 1,
    npmod=np
) -> np.ndarray:
    r"""Extract an eigenspace of the translation.

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

    num_steps = num_spins // shift
    if jphase not in list(range(num_steps)):
        raise ValueError('Invalid jphase value')

    op = translation(num_spins, shift=shift, npmod=npmod)
    eigval = np.exp(2.j * np.pi / num_steps * jphase)
    return get_eigenspace((op, eigval), basis, dim=2 ** num_spins, npmod=npmod)


def spin_flip(npmod=np) -> LinearOpFunction:
    r"""Return a function that applies a global spin flip :math:`\Pi_{\mathbb{Z}_2}`.

    Args:
        num_spins: Number of spins :math:`N`.

    Returns:
        A function that takes a basis matrix :math:`B` and computes :math:`\Pi_{\mathbb{Z}_2} B`.
    """
    def op(basis):
        """Flip the spins of the basis states."""
        # Spin flip of a state vector actually corresponds to a simple order reverse
        # [0000, 0001, 0010, ...] <-> [1111, 1110, 1101, ...]
        return npmod.array(basis[::-1, ...])

    if npmod is jnp:
        op = jax.jit(op)

    return op


def spin_flip_eigenspace(
    eigval: int,
    basis: Optional[np.ndarray] = None,
    num_spins: Optional[int] = None,
    npmod=np
) -> np.ndarray:
    r"""Extract the +1/-1 eigenspace of the global spin flip :math:`\Pi_{\mathbb{Z}_2}`.

    Args:
        eigval: +1 or -1.
        basis: The basis matrix :math:`B`.
        num_spins: Number of spins :math:`N`.

    Returns:
        A matrix whose columns form the orthonormal basis of the eigen-subspace.
    """
    if num_spins is None:
        num_spins = np.round(np.log2(basis.shape[0])).astype(int)

    if abs(eigval) != 1:
        raise ValueError('Invalid eigenvalue')

    op = spin_flip(npmod=npmod)
    return get_eigenspace((op, eigval), basis, dim=2 ** num_spins, npmod=npmod)
