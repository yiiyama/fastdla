"""Generators and symmetry projectors for Z2 lattice gauge theory HVA."""
from collections.abc import Sequence
from typing import Optional
import numpy as np
import jax
import jax.numpy as jnp
from fastdla.sparse_pauli_sum import SparsePauliSum, SparsePauliSumArray
from fastdla.generators.spin_chain import (
    translation_eigenspace,
    parity_eigenspace,
    spin_flip_eigenspace
)
from fastdla.algorithms.eigenspace import get_eigenspace


def z2lgt_physical_hva_generators(
    gauss_eigvals: Sequence[int],
    gauge_op: str = 'Z'
) -> SparsePauliSumArray:
    r"""Construct the generators of the HVA for the 1+1-dimensional Z2 Lattice gauge theory model
    with periodic boundary condition projected onto the physical-state basis for the given static
    charge configuration.

    The Hamiltonian of the 1+1d LGT with :math:`N_f` Dirac fermions (:math:`N_s = 2 N_f` lattice
    sites) for physical states in the charge sector defined by :math:`\{g_n\}_{n=0}^{N_s-1}` is

    .. math:: H = f H_{\mathrm{g}} + m H_{\mathrm{m}} + \frac{J}{2} H_{\mathrm{h}}

    where

    .. math::

        H_{\mathrm{g}} = \sum_{n=0}^{N_s-1} Z_{n,n+1}, \\
        H_{\mathrm{m}} = \sum_{n=0}^{N_s-1} (-1)^n g_n Z_{n-1,n} Z_{n,n+1}, \\
        H_{\mathrm{h}} = \sum_{n=0}^{N_s-1} (I - g_n g_{n+1} Z_{n-1,n} Z_{n+1,n+2}) X_{n,n+1}.

    In the above expressions, :math:`P_{n,n+1} (P=X,Z)` are operators acting on the link between
    sites :math:`n` and :math:`n+1`. The By the boundary condition, we identify site :math:`N_s`
    with site 0.

    We then assume a register of :math:`4 N_f` qubits in a ring topology, and map link :math:`n,n+1`
    to qubit :math:`n`. Under this mapping, we define the Hamiltonian variational ansatz (HVA) of
    this model as

    .. math::

        U(\vec{\theta}) = \prod_{l=0}^{L-1} e^{-i \theta_{l,0} H_{\mathrm{g}}}
                                            e^{-i \theta_{l,1} H_{\mathrm{m}}^{\mathrm{(even)}}}
                                            e^{-i \theta_{l,2} H_{\mathrm{m}}^{\mathrm{(odd)}}}
                                            e^{-i \theta_{l,3} H_{\mathrm{h}}^{\mathrm{(even)}}}
                                            e^{-i \theta_{l,4} H_{\mathrm{h}}^{\mathrm{(odd)}}}

    where :math:`L` is the number of repeated circuit layers, and even (odd) superscripts indicate
    taking the sum over even (odd) :math:`n` in the corresponding definition of the Hamiltonian
    term.

    The generators of the HVA

    .. math::

        \{ -iH_{\mathrm{g}}^{\mathrm{(even)}},
           -iH_{\mathrm{g}}^{\mathrm{(odd)}},
           -iH_{\mathrm{m}}^{\mathrm{(even)}},
           -iH_{\mathrm{m}}^{\mathrm{(odd)}}},
           -iH_{\mathrm{h}}^{\mathrm{(even)}},
           -iH_{\mathrm{h}}^{\mathrm{(odd)}}
        \}

    all commute with symmetry operators :math:`Q` (total charge) and :math:`T_2` (translation).
    :math:`Q` is defined as

    .. math::

        Q = \sum_{n=0}^{N_s-1} g_n Z_{n-1,n} Z_{n,n+1}.

    :math:`T_2` is defined as an operation that shifts site index by 2: :math:`n \to n+2`, and can
    be implemented with a series of qubit swap operations.

    Args:
        gauss_eigvals: Charge sector specification (eigenvalues {+1, -1} of :math:`g_n` for each
            site).

    Returns:
        Five generators of the HVA.
    """
    gauss_eigvals = np.asarray(gauss_eigvals)
    num_links = len(gauss_eigvals)
    generators = SparsePauliSumArray(num_qubits=num_links)
    flip_op = 'X' if gauge_op == 'Z' else 'Z'

    # Field term H_g(even)
    paulis = ['I' * (num_links - ilink - 1) + gauge_op + 'I' * ilink
              for ilink in range(0, num_links, 2)]
    generators.append(SparsePauliSum(paulis, -1.j * np.ones(num_links // 2)))

    # Field term H_g(odd)
    paulis = ['I' * (num_links - ilink - 1) + gauge_op + 'I' * ilink
              for ilink in range(1, num_links, 2)]
    generators.append(SparsePauliSum(paulis, -1.j * np.ones(num_links // 2)))

    # Mass term H_m(even)
    paulis = [gauge_op + 'I' * (num_links - 2) + gauge_op]
    paulis += ['I' * (num_links - ilink - 2) + gauge_op * 2 + 'I' * ilink
               for ilink in range(1, num_links - 1, 2)]
    generators.append(SparsePauliSum(paulis, -1.j * gauss_eigvals[::2]))

    # Mass term H_m(odd)
    paulis = ['I' * (num_links - ilink - 2) + gauge_op * 2 + 'I' * ilink
              for ilink in range(0, num_links, 2)]
    generators.append(SparsePauliSum(paulis, 1.j * gauss_eigvals[1::2]))

    # Hopping terms H_h(even)
    paulis = ['I' * (num_links - ilink - 1) + flip_op + 'I' * ilink
              for ilink in range(0, num_links, 2)]
    paulis.append(gauge_op + 'I' * (num_links - 3) + gauge_op + flip_op)
    paulis += ['I' * (num_links - ilink - 2) + gauge_op + flip_op + gauge_op + 'I' * (ilink - 1)
               for ilink in range(2, num_links, 2)]
    zxz_coeff = -0.5 * gauss_eigvals[::2] * gauss_eigvals[1::2]
    coeffs = np.concatenate([np.full(num_links // 2, 0.5), zxz_coeff])
    generators.append(SparsePauliSum(paulis, -1.j * coeffs))

    # Hopping terms H_h(odd)
    paulis = ['I' * (num_links - ilink - 1) + flip_op + 'I' * ilink
              for ilink in range(1, num_links, 2)]
    paulis += ['I' * (num_links - ilink - 2) + gauge_op + flip_op + gauge_op + 'I' * (ilink - 1)
               for ilink in range(1, num_links - 1, 2)]
    paulis.append('XZ' + 'I' * (num_links - 3) + 'Z')
    gauss_eigvals = np.asarray(gauss_eigvals)
    zxz_coeff = -0.5 * gauss_eigvals[1::2] * np.roll(gauss_eigvals[::2], -1)
    coeffs = np.concatenate([np.full(num_links // 2, 0.5), zxz_coeff])
    generators.append(SparsePauliSum(paulis, -1.j * coeffs))

    return generators


def z2lgt_physical_u1_charges(
    gauss_eigvals: Sequence[int],
    npmod=np
) -> np.ndarray:
    gauss_eigvals = npmod.asarray(gauss_eigvals)
    num_links = len(gauss_eigvals)

    idx = npmod.arange(2 ** num_links)
    bidx = ((idx[:, None] >> npmod.arange(num_links)[None, ::-1]) % 2).astype(np.uint8)
    # Z_{n-1,n} Z_{n,n+1}
    zz_parity = npmod.roll(bidx, -1, axis=1) ^ bidx
    return npmod.sum((1 - 2 * zz_parity) * gauss_eigvals[None, ::-1], axis=1)


def z2lgt_physical_u1_projector(
    gauss_eigvals: Sequence[int],
    charge: int
) -> SparsePauliSum:
    r"""Construct the projector for the U(1) symmetry for the given total charge.

    Total charge is defined as

    .. math::

        Q = \sum_{n=0}^{N_s-1} g_n Z_{n-1,n} Z_{n,n+1}.

        gauss_eigvals: Charge sector specification (eigenvalues {+1, -1} of :math:`g_n` for each
            site).
        total_charge: Unnormalized total charge (an eigenvalue of :math:`\sum_{n=0}^{N_s-1} Z_n`).
    """
    num_links = len(gauss_eigvals)
    charges = z2lgt_physical_u1_charges(gauss_eigvals)
    states = np.nonzero(np.equal(charges, charge))[0]
    # Binary representations of the indices
    states_binary = (states[:, None] >> np.arange(num_links)[None, ::-1]) % 2
    # |0><0|=1/2(I+Z), |1><1|=1/2(I-Z) -> Coefficients of I and Z for each binary digit
    # Example: [0, 1] -> [[1, 1], [1, -1]]
    states_iz = np.array([[1, 1], [1, -1]])[states_binary]
    # Take the kronecker products of the I/Z coefficients using einsum, then sum over the states to
    # arrive at the final sparse Pauli representation of the projector
    args = ()
    for isite in range(num_links):
        args += (states_iz[:, isite], [0, isite + 1])
    args += (list(range(num_links + 1)),)
    coeffs = np.sum(np.einsum(*args).reshape(states.shape[0], 2 ** num_links), axis=0)
    # Take only the nonzero Paulis
    pauli_indices = np.nonzero(coeffs)[0]
    coeffs = coeffs[pauli_indices] / (2 ** num_links)
    pauli_indices_bin = (pauli_indices[:, None] >> np.arange(num_links)[None, ::-1]) % 2
    paulis = [''.join('IZ'[b] for b in idx) for idx in pauli_indices_bin]
    return SparsePauliSum(paulis, coeffs)


def z2lgt_physical_symmetry_eigenspace(
    gauss_eigvals: Sequence[int],
    u1_charge: Optional[int] = None,
    z2_sign: Optional[int] = None,
    c_phase: Optional[int] = None,
    p_sign: Optional[int] = None,
    t2_momentum: Optional[int] = None,
    cp_sign: Optional[int] = None,
    basis: Optional[np.ndarray] = None,
    npmod=np
) -> np.ndarray:
    if u1_charge is not None:
        basis = z2lgt_physical_u1_eigenspace(gauss_eigvals, u1_charge, basis=basis, npmod=npmod)
    elif basis is None:
        basis = npmod.eye(2 ** len(gauss_eigvals), dtype=np.complex128)
    if z2_sign is not None:
        basis = z2lgt_physical_z2_eigenspace(gauss_eigvals, z2_sign, basis=basis, npmod=npmod)
    if c_phase is not None:
        basis = z2lgt_physical_c_eigenspace(gauss_eigvals, c_phase, basis=basis, npmod=npmod)
    if p_sign is not None:
        basis = z2lgt_physical_p_eigenspace(gauss_eigvals, p_sign, basis=basis, npmod=npmod)
    if t2_momentum is not None:
        basis = z2lgt_physical_t2_eigenspace(gauss_eigvals, t2_momentum, basis=basis, npmod=npmod)
    if cp_sign is not None:
        basis = z2lgt_physical_cp_eigenspace(gauss_eigvals, cp_sign, basis=basis, npmod=npmod)
    return basis


def z2lgt_physical_u1_eigenspace(
    gauss_eigvals: Sequence[int],
    charge: int,
    basis: Optional[np.ndarray] = None,
    npmod=np
) -> np.ndarray:
    r"""Extract the eigenspace of the U(1) symmetry for the given total charge.

    Total charge is defined as

    .. math::

        Q = \sum_{n=0}^{N_s-1} g_n Z_{n-1,n} Z_{n,n+1}.

    Args:
        gauss_eigvals: Charge sector specification (eigenvalues {+1, -1} of :math:`g_n` for each
            site).
        total_charge: Unnormalized total charge (an eigenvalue of :math:`\sum_{n=0}^{N_s-1} Z_n`).
        basis: The basis matrix :math:`B`.
        num_fermions: :math:`N_f`.

    Returns:
        A matrix whose columns form the orthonormal basis of the eigen-subspace.
    """
    num_links = len(gauss_eigvals)
    charges = z2lgt_physical_u1_charges(gauss_eigvals, npmod=npmod)

    if basis is None:
        # Directly extract one-hot vectors
        eigen_idx = npmod.nonzero(npmod.equal(charges, charge))[0]
        if npmod is np:
            subspace = np.zeros((2 ** num_links, eigen_idx.shape[0]), dtype=np.complex128)
            subspace[eigen_idx, np.arange(eigen_idx.shape[0])] = 1.
        else:
            subspace = jax.nn.one_hot(eigen_idx, 2 ** num_links, dtype=np.complex128).T
    else:
        def op(_basis):
            return _basis * npmod.not_equal(charges, charge).astype(int)[:, None]

        if npmod is jnp:
            op = jax.jit(op)

        subspace = get_eigenspace(op, basis, dim=2 ** num_links, npmod=npmod)

    return subspace


def z2lgt_physical_z2_eigenspace(
    gauss_eigvals: Sequence[int],
    sign: int,
    basis: Optional[np.ndarray] = None,
    npmod=np
) -> np.ndarray:
    return spin_flip_eigenspace(sign, basis=basis, num_spins=len(gauss_eigvals), npmod=npmod)


def z2lgt_physical_c_eigenspace(
    gauss_eigvals: Sequence[int],
    phase: int,
    basis: Optional[np.ndarray] = None,
    npmod=np
) -> np.ndarray:
    r"""Extract the eigenspace of C for the given eigenphase index.
    Args:
        gauss_eigvals: Charge sector specification (eigenvalues {+1, -1} of :math:`g_n` for each
            site).
        c_phase: Index :math:`c` of the eigenvalue :math:`e^{1 \pi i c / N_s}` of
            :math:`\mathcal{C}`.
        basis: The basis matrix :math:`B`.

    Returns:
        A matrix whose columns form the orthonormal basis of the eigen-subspace.
    """
    return translation_eigenspace(phase, basis=basis, num_spins=len(gauss_eigvals), npmod=npmod)


def z2lgt_physical_p_eigenspace(
    gauss_eigvals: Sequence[int],
    sign: int,
    basis: Optional[np.ndarray] = None,
    npmod=np
) -> np.ndarray:
    return parity_eigenspace(sign, basis=basis, num_spins=len(gauss_eigvals), reflect_about=(-1, 0),
                             npmod=npmod)


def z2lgt_physical_t2_eigenspace(
    gauss_eigvals: Sequence[int],
    momentum: int,
    basis: Optional[np.ndarray] = None,
    npmod=np
) -> np.ndarray:
    return translation_eigenspace(momentum, basis=basis, num_spins=len(gauss_eigvals), shift=2,
                                  npmod=npmod)


def z2lgt_physical_cp_eigenspace(
    gauss_eigvals: Sequence[int],
    sign: int,
    basis: Optional[np.ndarray] = None,
    npmod=np
) -> np.ndarray:
    num_links = len(gauss_eigvals)

    def cp_kernel(basis):
        transformed = basis.reshape((2,) * num_links + (-1,))
        # P
        transformed = npmod.moveaxis(transformed, np.arange(num_links), np.arange(num_links)[::-1])
        # C
        src = np.arange(num_links)
        dest = np.roll(np.arange(num_links), -1)
        transformed = npmod.moveaxis(transformed, src, dest)
        # transformed is the OB in the get_eigenspace doc
        return transformed.reshape((2 ** num_links, -1)) - sign * basis

    if npmod is jnp:
        cp_kernel = jax.jit(cp_kernel)

    return get_eigenspace(cp_kernel, basis=basis, dim=2 ** num_links, npmod=npmod)
