"""Generators and symmetry projectors for Z2 lattice gauge theory HVA."""
from collections.abc import Sequence
from typing import Optional
import numpy as np
import jax
import jax.numpy as jnp
from fastdla.sparse_pauli_sum import SparsePauliSum, SparsePauliSumArray
from fastdla.generators.spin_chain import translation_eigenspace, parity_eigenspace
from fastdla.linalg.eigenspace import get_eigenspace


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

        \{ iH_{\mathrm{g}}}, iH_{\mathrm{m}}^{\mathrm{(even)}}}, iH_{\mathrm{m}}^{\mathrm{(odd)}}},
           iH_{\mathrm{h}}^{\mathrm{(even)}}}, iH_{\mathrm{h}}^{\mathrm{(odd)}}} \}

    all commute with symmetry operators :math:`Q` (total charge) and :math:`T_2` (translation).
    :math:`Q` is defined as

    .. math::

        Q = \frac{1}{N_s} \sum_{n=0}^{N_s-1} g_n Z_{n-1,n} Z_{n,n+1}.

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

    # Field term H_g
    paulis = ['I' * (num_links - ilink - 1) + gauge_op + 'I' * ilink
              for ilink in range(num_links)]
    generators.append(SparsePauliSum(paulis, 1.j * np.ones(num_links)))

    # Mass term H_m(even)
    paulis = [gauge_op + 'I' * (num_links - 2) + gauge_op]
    paulis += ['I' * (num_links - ilink - 2) + gauge_op * 2 + 'I' * ilink
               for ilink in range(1, num_links - 1, 2)]
    generators.append(SparsePauliSum(paulis, 1.j * gauss_eigvals[::2]))

    # Mass term H_m(odd)
    paulis = ['I' * (num_links - ilink - 2) + gauge_op * 2 + 'I' * ilink
              for ilink in range(0, num_links, 2)]
    generators.append(SparsePauliSum(paulis, -1.j * gauss_eigvals[1::2]))

    # Hopping terms H_h(even)
    paulis = ['I' * (num_links - ilink - 1) + flip_op + 'I' * ilink
              for ilink in range(0, num_links, 2)]
    paulis.append(gauge_op + 'I' * (num_links - 3) + gauge_op + flip_op)
    paulis += ['I' * (num_links - ilink - 2) + gauge_op + flip_op + gauge_op + 'I' * (ilink - 1)
               for ilink in range(2, num_links, 2)]
    zxz_coeff = -0.5 * gauss_eigvals[::2] * gauss_eigvals[1::2]
    coeffs = np.concatenate([np.full(num_links // 2, 0.5), zxz_coeff])
    generators.append(SparsePauliSum(paulis, 1.j * coeffs))

    # Hopping terms H_h(odd)
    paulis = ['I' * (num_links - ilink - 1) + flip_op + 'I' * ilink
              for ilink in range(1, num_links, 2)]
    paulis += ['I' * (num_links - ilink - 2) + gauge_op + flip_op + gauge_op + 'I' * (ilink - 1)
               for ilink in range(1, num_links - 1, 2)]
    paulis.append('XZ' + 'I' * (num_links - 3) + 'Z')
    gauss_eigvals = np.asarray(gauss_eigvals)
    zxz_coeff = -0.5 * gauss_eigvals[1::2] * np.roll(gauss_eigvals[::2], -1)
    coeffs = np.concatenate([np.full(num_links // 2, 0.5), zxz_coeff])
    generators.append(SparsePauliSum(paulis, 1.j * coeffs))

    return generators


def z2lgt_physical_dense_u1_eigenspace(
    gauss_eigvals: Sequence[int],
    total_charge: int,
    basis: Optional[np.ndarray] = None,
    npmod=np
) -> np.ndarray:
    r"""Extract the eigenspace of the U(1) symmetry for the given total charge.

    Args:
        gauss_eigvals: Charge sector specification (eigenvalues {+1, -1} of :math:`g_n` for each
            site).
        total_charge: Unnormalized total charge (an eigenvalue of :math:`\sum_{n=0}^{N_s-1} Z_n`).
        basis: The basis matrix :math:`B`.
        num_fermions: :math:`N_f`.

    Returns:
        A matrix whose columns form the orthonormal basis of the eigen-subspace.
    """
    gauss_eigvals = np.asarray(gauss_eigvals)
    num_links = len(gauss_eigvals)

    idx = npmod.arange(2 ** num_links)
    bidx = ((idx[:, None] >> npmod.arange(num_links)[None, ::-1]) % 2).astype(np.uint8)
    charges = npmod.zeros(2 ** num_links, dtype=int)
    for ilink, gn in enumerate(gauss_eigvals):
        parity = bidx[:, num_links - 1 - ilink] ^ bidx[:, (-ilink) % num_links]
        charges += (1 - 2 * parity) * gn

    if basis is None:
        # Directly extract one-hot vectors
        target_charge_states = npmod.nonzero(npmod.equal(charges, total_charge))[0]
        if npmod is np:
            subspace = np.zeros((2 ** num_links, idx.shape[0]), dtype=np.complex128)
            subspace[target_charge_states, np.arange(target_charge_states.shape[0])] = 1.
        else:
            subspace = jax.nn.one_hot(target_charge_states, 2 ** num_links).T
    else:
        def op(_basis):
            return _basis * npmod.not_equal(charges, total_charge).astype(int)[:, None]

        if npmod is jnp:
            op = jax.jit(op)

        subspace = get_eigenspace(op, basis, dim=2 ** num_links, npmod=npmod)

    return subspace


def z2lgt_physical_dense_c_eigenspace(
    gauss_eigvals: Sequence[int],
    c_phase: int,
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
    return translation_eigenspace(c_phase, basis=basis, num_spins=len(gauss_eigvals),
                                  npmod=npmod)


def z2lgt_physical_dense_p_eigenspace(
    gauss_eigvals: Sequence[int],
    parity: int,
    basis: Optional[np.ndarray] = None,
    npmod=np
) -> np.ndarray:
    return parity_eigenspace(parity, basis=basis, num_spins=len(gauss_eigvals), npmod=npmod)


def z2lgt_physical_dense_t2_eigenspace(
    gauss_eigvals: Sequence[int],
    t2_momentum: int,
    basis: Optional[np.ndarray] = None,
    npmod=np
) -> np.ndarray:
    return translation_eigenspace(t2_momentum, basis=basis, num_spins=len(gauss_eigvals), shift=2,
                                  npmod=npmod)


def z2lgt_physical_dense_cp_eigenspace(
    gauss_eigvals: Sequence[int],
    cp_parity: int,
    basis: Optional[np.ndarray] = None,
    npmod=np
) -> np.ndarray:
    num_links = len(gauss_eigvals)

    def cp_kernel(basis):
        idx = npmod.arange(2 ** num_links)
        bidx = (idx[:, None] >> npmod.arange(num_links)[None, ::-1]) % 2
        dest_bidx = npmod.roll(bidx[:, ::-1], 1, axis=1)
        dest_idx = npmod.sum(dest_bidx * (1 << npmod.arange(num_links)[::-1]), axis=1)
        return basis[dest_idx, :] - cp_parity * basis

    if npmod is jnp:
        cp_kernel = jax.jit(cp_kernel)

    return get_eigenspace(cp_kernel, basis=basis, dim=2 ** num_links, npmod=npmod)
