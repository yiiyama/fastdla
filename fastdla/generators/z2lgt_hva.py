"""Generators and symmetry projectors for Z2 lattice gauge theory HVA."""
from collections.abc import Sequence
from typing import Optional
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from ..pauli import PAULIS
from ..sparse_pauli_sum import SparsePauliSum, SparsePauliSumArray
from .spin_chain import translation, translation_eigenspace
from ..linalg.eigenspace import LinearOpFunction, get_eigenspace


def z2lgt_hva_generators(num_fermions: int, gauge_op: str = 'X') -> SparsePauliSumArray:
    r"""Construct the generators of the HVA for the 1+1-dimensional Z2 Lattice gauge theory model
    with periodic boundary condition.

    The Hamiltonian of the 1+1d LGT with :math:`N_f` Dirac fermions (:math:`N_s = 2 N_f` lattice
    sites) and a periodic boundary condition is given by

    .. math:: H = f H_{\mathrm{g}} + m H_{\mathrm{m}} + \frac{J}{2} H_{\mathrm{h}}

    where

    .. math::

        H_{\mathrm{g}} = \sum_{n=0}^{N_s-1} X_{n,n+1}, \\
        H_{\mathrm{m}} = \sum_{n=0}^{N_s-1} (-1)^n Z_n, \\
        H_{\mathrm{h}} = \sum_{n=0}^{N_s-1} (X_n Z_{n,n+1} X_{n+1} + Y_n Z_{n,n+1} Y_{n+1}).

    In the above expressions, :math:`P_n (P=X,Y,Z)` are the Pauli operators acting on site
    :math:`n`, and :math:`P_{n,n+1} (P=X,Z)` are those acting on the link between sites :math:`n`
    and :math:`n+1`. By the boundary condition, we identify site :math:`N_s` with site 0.

    We then assume a register of :math:`4 N_f` qubits in a ring topology, and map site :math:`n` and
    link :math:`n,n+1` to qubits :math:`2n` and :math:`2n+1`, respectively. Under this mapping, we
    define the Hamiltonian variational ansatz (HVA) of this model as

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

        \{ H_{\mathrm{g}}}, H_{\mathrm{m}}^{\mathrm{(even)}}}, H_{\mathrm{m}}^{\mathrm{(odd)}}},
           H_{\mathrm{h}}^{\mathrm{(even)}}}, H_{\mathrm{h}}^{\mathrm{(odd)}}} \}

    all commute with symmetry operators :math:`\{G_n\}_{n=0}^{N_s-1}` (Gauss's law), :math:`Q`
    (total charge), and :math:`T_2` (translation). The definitions of :math:`G_n` and :math:`Q` are
    given in terms of Pauli operators as

    .. math::

        G_n = X_{n-1,n} Z_n X_{n,n+1}, \\
        Q = \frac{1}{N_s} \sum_{n=0}^{N_s-1} Z_n.

    :math:`T_2` is defined as an operation that shifts site index by 2: :math:`n \to n+2` (qubit
    index by 4), and can be implemented with a series of qubit swap operations.

    Args:
        num_fermions: Number of fermions :math:`N_f`.

    Returns:
        Five generators of the HVA.
    """
    num_qubits = 4 * num_fermions

    generators = SparsePauliSumArray(num_qubits=num_qubits)

    # Field term H_g
    strings = ['I' * (num_qubits - iq - 1) + gauge_op + 'I' * iq
               for iq in range(1, num_qubits, 2)]
    coeffs = np.ones(len(strings)) / np.sqrt(len(strings))
    generators.append(SparsePauliSum(strings, coeffs))

    # Mass terms H_m (even and odd)
    for parity in [0, 1]:
        strings = ['I' * (num_qubits - isite * 2 - 1) + 'Z' + 'I' * (isite * 2)
                   for isite in range(parity, 2 * num_fermions, 2)]
        coeffs = np.ones(len(strings)) / np.sqrt(len(strings))
        generators.append(SparsePauliSum(strings, coeffs))

    # Hopping terms H_h (even and odd)
    link_op = 'Z' if gauge_op == 'X' else 'X'
    for parity in [0, 1]:
        strings = []
        for isite in range(parity, 2 * num_fermions, 2):
            for site_op in ['X', 'Y']:
                paulis_reverse = ['I'] * num_qubits
                paulis_reverse[isite * 2] = site_op
                paulis_reverse[isite * 2 + 1] = link_op
                paulis_reverse[(isite * 2 + 2) % num_qubits] = site_op
                strings.append(''.join(paulis_reverse[::-1]))
        coeffs = np.ones(len(strings)) / np.sqrt(len(strings))
        generators.append(SparsePauliSum(strings, coeffs))

    return generators


def z2lgt_gauss_local_projector(
    num_fermions: int,
    isite: int,
    eigval: int,
    gauge_op: str = 'X'
) -> SparsePauliSum:
    r"""Construct the Gauss's law projector for the Z2 LGT model for a single matter site.

    Physical states of the Z2 LGT model must be eigenstates of

    .. math::

        G_n = X_{n-1,n}Z_{n}X_{n1,n+1}.

    Given an eigenvalue for each G_n, we can construct a projector to its subspace as a sum of Pauli
    ops.
    """
    if abs(eigval) != 1:
        raise ValueError('Charge value must be +-1')

    num_qubits = 4 * num_fermions

    if eigval > 0:
        # +0+ -1+ -0- +1- with +/-=1/2(I±X) and 0/1=1/2(I±Z)
        # From
        # np.sum([
        #     np.kron(np.kron([1, 1], [1, 1]), [1, 1]),
        #     np.kron(np.kron([1, -1], [1, -1]), [1, 1]),
        #     np.kron(np.kron([1, -1], [1, 1]), [1, -1]),
        #     np.kron(np.kron([1, 1], [1, -1]), [1, -1])
        # ]) / 8
        # we get array([0.5, 0, 0, 0, 0, 0, 0, 0.5]) -> 1/2(III + XZX)

        coeffs = [0.5, 0.5]
    else:
        # -1- +0- +1+ -0+ with +/-=1/2(I±X) and 0/1=1/2(I±Z)
        # From
        # np.sum([
        #     np.kron(np.kron([1, 1], [1, 1]), [1, 1]),
        #     np.kron(np.kron([1, -1], [1, -1]), [1, 1]),
        #     np.kron(np.kron([1, -1], [1, 1]), [1, -1]),
        #     np.kron(np.kron([1, 1], [1, -1]), [1, -1])
        # ]) / 8
        # we get array([0.5, 0, 0, 0, 0, 0, 0, -0.5]) -> 1/2(III - XZX)
        coeffs = [0.5, -0.5]

    strings = ['I' * num_qubits]
    paulis_reverse = ['I'] * num_qubits
    paulis_reverse[isite * 2] = 'Z'
    paulis_reverse[(isite * 2 - 1) % num_qubits] = gauge_op
    paulis_reverse[(isite * 2 + 1) % num_qubits] = gauge_op
    strings.append(''.join(paulis_reverse[::-1]))

    return SparsePauliSum(strings, coeffs)


def z2lgt_gauss_projector(eigvals: Sequence[int], gauge_op: str = 'X') -> SparsePauliSum:
    r"""Construct the Gauss's law projector for the Z2 LGT model.

    Physical states of the Z2 LGT model must be eigenstates of

    .. math::

        G_n = X_{n-1,n}Z_{n}X_{n1,n+1}.

    Given an eigenvalue for each G_n, we can construct a projector to its subspace as a sum of Pauli
    ops. The overall projector will then be a product of all such projectors.
    """
    if (num_sites := len(eigvals)) % 2 or not all(abs(ev) == 1 for ev in eigvals):
        raise ValueError('There must be an even number of charges with values +-1')

    num_fermions = num_sites // 2
    num_qubits = 2 * num_sites

    projector = SparsePauliSum('I' * num_qubits, 1.)
    for isite, ev in enumerate(eigvals):
        projector = projector @ z2lgt_gauss_local_projector(num_fermions, isite, ev,
                                                            gauge_op=gauge_op)

    return projector


def z2lgt_u1_projector(num_fermions: int, charge: int) -> SparsePauliSum:
    r"""Construct the charge conservation law projector for the Z2 LGT model.

    Physical states of the Z2 LGT model must be eigenstates of

    .. math::

        Q = \frac{1}{N_s} \sum_{n=0}^{N_s-1} Z_n.

    For a fermion number :math:`N_f` and an overall charge :math:`q \in [-2N_f,...,2N_f]`, we have a
    :math:`{}_{2N_f} C_(N_f+q/2)`-dimensional eigenspace.
    """
    if abs(charge) > (num_sites := 2 * num_fermions) or charge % 2 != 0:
        raise ValueError('Invalid charge value')

    # Diagonals=eigenvalues of the symmetry generator
    eigvals = np.zeros((2,) * num_sites, dtype=int)
    z = np.array([1, -1])
    for isite in range(num_sites):
        eigvals += np.expand_dims(z, tuple(range(isite)) + tuple(range(isite + 1, num_sites)))
    eigvals = eigvals.reshape(-1)
    # State indices with the given charge
    states = np.nonzero(eigvals == charge)[0]
    # Binary representations of the indices
    states_binary = (states[:, None] >> np.arange(num_sites)[None, ::-1]) % 2
    # |0><0|=1/2(I+Z), |1><1|=1/2(I-Z) -> Coefficients of I and Z for each binary digit
    # Example: [0, 1] -> [[1, 1], [1, -1]]
    states_iz = np.array([[1, 1], [1, -1]])[states_binary]
    # Take the kronecker products of the I/Z coefficients using einsum, then sum over the states to
    # arrive at the final sparse Pauli representation of the projector
    args = ()
    for isite in range(num_sites):
        args += (states_iz[:, isite], [0, isite + 1])
    args += (list(range(num_sites + 1)),)
    coeffs = np.sum(np.einsum(*args).reshape(states.shape[0], 2 ** num_sites), axis=0)
    # Take only the nonzero Paulis
    pauli_indices = np.nonzero(coeffs)[0]
    coeffs = coeffs[pauli_indices] / (2 ** num_sites)
    paulis = []
    for idx in pauli_indices:
        idx_bin = np.zeros((num_sites, 2), dtype=int)
        idx_bin[:, 1] = (idx >> np.arange(num_sites)[::-1]) % 2
        paulis.append(''.join('IZ'[i] for i in idx_bin.reshape(-1)))

    return SparsePauliSum(paulis, coeffs)


def z2lgt_translation_projector(num_fermions: int, jphase: int) -> SparsePauliSum:
    """Construct the translation projector for the Z2 LGT model.

    The Z2 LGT HVA generators commute with the translation operator :math:`T_2`, which shifts state
    indices by 1 fermion unit (2 sites, 4 qubits). Therefore, if the initial state of a VQA is a
    translation eigenstate, the variational evolution of the state stays within the same eigenspace.
    Since it is quite nontrivial to directly express the translation operator eigenspace projectors
    in terms of Pauli product sums, we first construct the projectors as dense matrices, then
    compute the inner products with Paulis.
    """
    if jphase < 0 or jphase >= num_fermions:
        raise ValueError('Invalid iroot value')

    num_qubits = 4 * num_fermions

    # Construct the translation eigenstates from computational basis states
    # Translation eigenstate can be constructed from an arbitrary state |ψ> by
    #    |ψ_Tj> = sum_k e^{-2πi/N_f jk} T^k |ψ>
    # with an appropriate normalization.
    # This will be the full collection of column eigenvectors
    eigenstates = np.eye(2 ** num_qubits, dtype=np.complex128) / num_fermions
    powers = 1 << np.arange(num_qubits)[::-1]
    indices = np.arange(2 ** num_qubits)
    # Binary indices [[0, ..., 0, 0], [0, ..., 0, 1], ...]
    indices_bin = (indices[:, None] // powers[None, :]) % 2
    for shift in range(1, num_fermions):
        # Roll the binaries by 4 and recombine them into index integers
        shifted_indices = np.sum(np.roll(indices_bin, 4 * shift, axis=1) * powers[None, :], axis=1)
        # e^{-2πi/N_f jk}
        phase_factor = np.exp(-2.j * np.pi * jphase * shift / num_fermions)
        eigenstates[shifted_indices, indices] += phase_factor / num_fermions
    # From eigenstates = (|u_0>, |u_1>, ...) compute the projector = sum_j |u_j><u_j|
    # Each computational basis state is a part of n-cycle (n=prime factor of N_f).
    # After the above procedure, there are N_f/n columns that share the same n-cycle.
    # Since each column is a sum of N_f terms uniformly normalized by N_f, the summation results in
    # a proper orthonormal projector.
    projector = eigenstates @ eigenstates.conjugate().T

    # Pauli decomposition through iterative partial traces
    projector = projector.reshape((2,) * (2 * num_qubits))
    # (p, row, column) . (row_{N_q-1}, ..., row_{0}, col_{N_q-1}, ..., col_{0})
    # -> (p, row, column) . (p_{N_q-1}, row_{N_q-2}, ..., row_{0}, col_{N_q-2}, ..., col_{0})
    # -> (p, row, column) . (p_{N_q-2}, p_{N_q-1}, ..., row_{0}, col_{N_q-3}, ..., col_{0})
    # -> ...
    # -> (p_{0}, p_{1}, ..., p_{N_q-1})
    for iq in range(num_qubits):
        projector = np.tensordot(PAULIS, projector, [[1, 2], [num_qubits, iq]]) / 2.

    # Sparsify
    projector = np.reshape(np.transpose(projector), -1)
    indices = np.nonzero(projector)[0]
    coeffs = projector[indices]

    return SparsePauliSum(indices, coeffs, num_qubits=num_qubits)


def z2lgt_symmetry_eigenspace(
    gauss_eigvals: Sequence[int],
    u1_total_charge: Optional[int] = None,
    t_jphase: Optional[int] = None,
    gauge_op: str = 'X',
    npmod=np
) -> np.ndarray:
    """Construct a full symmetry projector for the Z2 LGT model.

    The construction follows the projection algorithm for multiple symmetries proposed by LN. See
    the docstring of individual projectors for details.

    Returns a (pu, 2**nq) matrix.
    """
    basis = z2lgt_dense_gauss_eigenspace(gauss_eigvals, gauge_op=gauge_op, npmod=npmod)
    if u1_total_charge is not None:
        basis = z2lgt_dense_u1_eigenspace(u1_total_charge, basis, npmod=npmod)
    if t_jphase is not None:
        basis = z2lgt_dense_translation_eigenspace(t_jphase, basis, npmod=npmod)

    return basis


def z2lgt_dense_gauss_eigenspace(
    gauss_eigvals: Sequence[int],
    gauge_op: str = 'X',
    npmod=np
) -> np.ndarray:
    r"""Get the eigenspace basis of Gauss's law operators.

    The construction follows the projection algorithm for multiple symmetries proposed by LN. To
    avoid constructing the full :math:`2^{n_q} \times 2^{n_q}` matrix, we compose the final
    projector from local-term projectors, extending the dimension as necessary.

    **Algorithm:**

    Let the number of matter sites be :math:`M`, qubits be indexed from right to left, and qubit 0
    correspond to matter site 0.

    Start by finding the eigenspace of the left-most (:math:`M-1`) Gauss law operator
    :math:`X_{M-2,M-1}Z_{M-1}X_{M-1,0}` corresponding to `gauss_eigvals[-1]`. Construct a projector
    :math:`P(M-1)` with shape :math:`[d(M-1)=8, p(M-1)=4]`.

    For a matter site :math:`M-j-1 \; (j=1,...,M-2)`, extend the dimensions of the current projector
    :math:`P(M-j)` to :math:`4d(M-j)`, then project the Gauss law operator onto the subspace to
    obtain

    .. math::

        R(M-j-1) = [I \otimes I \otimes P(M-j)^{\dagger}] [X_{M-j-2,M-j-1}Z_{M-j-1}X_{M-j-1,M-j}
                                                           \otimes I^{\otimes 2j}]
                   [I \otimes I \otimes P(M-j)]

    (shape :math:`[4p(M-j), 4p(M-j)]`). Assemble the eigenvectors of :math:`R(M-j-1)` with
    eigenvalue `gauss_eigvals[M-j-1]` into :math:`S(M-j-1)=(v_0 \; \dots \; v_{2p(M-j)-1})`. The
    current projector :math:`I \otimes I \otimes P(M-j)` projected onto :math:`S(M-j-1)` is the next
    projector

    .. math::

        P(M-j-1) = [I \otimes I \otimes P(M-j)] S(M-j-1)

    (shape :math:`[d(M-j-1)=4d(M-j), p(M-j-1)=2p(M-j)]`).

    For the matter site 0, extend the dimension of :math:`P(1)` only by 2. The "local" projector
    actually acts on the leftmost qubit as well as the rightmost two. :math:`R(0)` and
    :math:`S(0)` have shapes :math:`[2p(1), 2p(1)]` and :math:`[2p(1), p(1)]` and the final
    projector will be :math:`[d(0)=2d(1)=2^{2M}, p(0)=p(1)=2^M]`.

    Args:
        gauss_eigvals: Sequence (length :math:`2N_f`) of eigenvalues (±1) of :math:`G_n`.

    Returns:
        An array of shape ``(2**N_q, 2**N_s)``, which represents the basis column vectors of the
        eigenspace.
    """
    if len(gauss_eigvals) % 2 or not all(abs(ev) == 1 for ev in gauss_eigvals):
        raise ValueError('There must be an even number of charges with values +-1')

    num_sites = len(gauss_eigvals)

    if gauge_op == 'X':
        pauliz = np.diagflat([1.+0.j, -1.+0.j])
        paulix = np.array([[0., 1.+0.j], [1.+0.j, 0.]])
        gauss_op = np.kron(paulix, np.kron(pauliz, paulix))
    else:
        zdiag = np.array([1.+0.j, -1.+0.j])
        gauss_op = np.diagflat(np.kron(zdiag, np.kron(zdiag, zdiag)))

    # Gauss's law projector
    # Start from the leftmost link-site-link and iteratively construct the full-size projector
    for isite_r, ev in enumerate(gauss_eigvals[::-1]):
        previous_d = 2 ** (2 * isite_r + 1)
        previous_p = 2 ** (isite_r + 1)

        if isite_r == 0:
            eigvals, eigvecs = np.linalg.eigh(gauss_op)
            basis = eigvecs[:, np.isclose(eigvals, ev)]
        elif isite_r < num_sites - 1:
            # Reshape and einsum instead of expanding the dimension of the projector and then
            # performing the matrix multiplication
            # [P(M-j)†⊗I⊗I] [I..I⊗XZX] [P(M-j)⊗I⊗I] (reverse order wrt the docstring)
            proj = basis.reshape((2 ** (2 * isite_r), 2, previous_p))
            projected_gauss = npmod.einsum('ijk,jlmn,imo->klon',
                                           proj.conjugate(),
                                           gauss_op.reshape((2, 4, 2, 4)),
                                           proj).reshape((previous_p * 4,) * 2)
            eigvals, eigvecs = npmod.linalg.eigh(projected_gauss)
            kwargs = {} if npmod is np else {'size': previous_p * 2}
            indices = npmod.nonzero(npmod.isclose(eigvals, ev), **kwargs)[0]
            subspace = eigvecs[:, indices].reshape(previous_p, 4, previous_p * 2)
            basis = npmod.einsum('ij,jkl->ikl', basis, subspace).reshape(previous_d * 4,
                                                                         previous_p * 2)
        else:
            # Transpose the rightmost X to position 0 simultaneously with matrix multiplication
            # [P(M-j)†⊗I] [X⊗I..I⊗XZ] [P(M-j)⊗I] (reverse order wrt the docstring)
            proj = basis.reshape((2, 2 ** (2 * isite_r - 1), 2, previous_p))
            projected_gauss = npmod.einsum('ijkl,kminop,pjnq->lmqo',
                                           proj.conjugate(),
                                           gauss_op.reshape((2,) * 6),
                                           proj).reshape((previous_p * 2,) * 2)
            eigvals, eigvecs = npmod.linalg.eigh(projected_gauss)
            kwargs = {} if npmod is np else {'size': previous_p}
            indices = npmod.nonzero(npmod.isclose(eigvals, ev), **kwargs)[0]
            subspace = eigvecs[:, indices].reshape(previous_p, 2, previous_p)
            basis = npmod.einsum('ij,jkl->ikl', basis, subspace).reshape(previous_d * 2, previous_p)

    return basis


# pylint: disable-next=invalid-name
_z2lgt_jnp_gauss_eigenspace = jax.jit(partial(z2lgt_dense_gauss_eigenspace, npmod=jnp),
                                      static_argnums=[0], static_argnames=['gauge_op'])


def z2lgt_jnp_gauss_eigenspace(eigvals, gauge_op='X'):
    # pylint: disable-next=not-callable
    return _z2lgt_jnp_gauss_eigenspace(tuple(eigvals), gauge_op=gauge_op)


def z2lgt_dense_u1_projection(
    total_charge: int,
    num_fermions: int,
    npmod=np
) -> LinearOpFunction:
    r"""Return a function that projects out the eigensubspace of the total U(1) charge for the
    given eigenvalue.

    If :math:`P_{\lambda}` is the projector to the eigensubspace for eigenvalue :math:`\lambda`,
    :math:`P_{\lambda} - I` is the (negative) anti-projector. It also is a singular operator used in
    the SVD eigenspace extraction algorithm (for the eigenspace of :math:`P_{\lambda}` with
    eigenvalue 1).

    Args:
        total_charge: Unnormalized total charge (an eigenvalue of :math:`\sum_{n=0}^{N_s-1} Z_n`).
        num_fermions: :math:`N_f`.

    Returns:
        A function that takes a basis matrix :math:`B` as an argument and projects out the
        eigensubspace corresponding to eigenvalue :math:`q / N_s`, where :math:`q` is the given
        total charge.
    """
    num_sites = 2 * num_fermions
    num_qubits = 2 * num_sites

    if abs(total_charge) > num_sites or total_charge % 2 != 0:
        raise ValueError('Invalid charge value')

    # Calculate the total charge for all site charge configurations
    eigvals = npmod.zeros((2,) * num_sites, dtype=int)
    z = npmod.array([1, -1])
    for isite in range(num_sites):
        eigvals += npmod.expand_dims(z, tuple(range(isite)) + tuple(range(isite + 1, num_sites)))
    eigvals = eigvals.reshape(-1)
    # Site charge configurations that correspond to the target total charge
    target_charge_states = npmod.nonzero(npmod.equal(eigvals, total_charge))[0]

    def op(basis):
        """Return the singular matrix."""
        transformed = npmod.asarray(basis)
        transformed = transformed.reshape((2,) * num_qubits + (-1,))
        # Move the site axes to the front and reserialize
        transformed = npmod.moveaxis(transformed, tuple(range(1, num_qubits, 2)),
                                     tuple(range(num_sites)))
        transformed = npmod.reshape(transformed, (2 ** num_sites, -1))
        # Project out states with charge configurations giving the target total charge
        mask = np.ones(2 ** num_sites)
        mask[target_charge_states] = 0.
        transformed *= mask[:, None]
        # Revert the axes
        transformed = npmod.reshape(transformed, (2,) * num_qubits + (-1,))
        transformed = npmod.moveaxis(transformed, tuple(range(num_sites)),
                                     tuple(range(1, num_qubits, 2)))
        transformed = npmod.reshape(transformed, (2 ** num_qubits, -1))
        return transformed

    if npmod is jnp:
        op = jax.jit(op)

    return op


def z2lgt_dense_u1_eigenspace(
    total_charge: int,
    basis: Optional[np.ndarray] = None,
    num_fermions: Optional[int] = None,
    npmod=np
) -> np.ndarray:
    r"""Extract the eigenspace of the U(1) symmetry for the given total charge.

    Args:
        total_charge: Unnormalized total charge (an eigenvalue of :math:`\sum_{n=0}^{N_s-1} Z_n`).
        basis: The basis matrix :math:`B`.
        num_fermions: :math:`N_f`.

    Returns:
        A matrix whose columns form the orthonormal basis of the eigen-subspace.
    """
    if num_fermions is None:
        num_fermions = np.round(np.log2(basis.shape[0])).astype(int) // 4
    op = z2lgt_dense_u1_projection(total_charge, num_fermions, npmod=npmod)
    return get_eigenspace(op, basis, dim=2 ** (num_fermions * 4), npmod=npmod)


def z2lgt_dense_translation(
    num_fermions: int,
    npmod=np
) -> tuple[LinearOpFunction, complex]:
    r"""Return a function that applies the translation :math:`T_2` to state vectors.

    Args:
        num_fermions: :math:`N_f`.

    Returns:
        A function that takes a basis matrix :math:`B` and computes :math:`T_2 B` and the eigenvalue
        :math:`e^{2\pi i j/N_f}`.
    """
    num_qubits = num_fermions * 4
    return translation(num_qubits, shift=4, npmod=npmod)


def z2lgt_dense_translation_eigenspace(
    jphase: int,
    basis: Optional[np.ndarray] = None,
    num_fermions: Optional[int] = None,
    npmod=np
) -> np.ndarray:
    r"""Extract the :math:`j`th eigenspace of the translation :math:`T_2`.

    Args:
        jphase: Integer :math:`j` of the :math:`T_2` eigenvalue :math:`e^{2\pi i j/N_f}`.
        basis: The basis matrix :math:`B`.
        num_fermions: :math:`N_f`.

    Returns:
        A matrix whose columns form the orthonormal basis of the eigen-subspace.
    """
    if num_fermions is None:
        num_fermions = np.round(np.log2(basis.shape[0])).astype(int) // 4
    return translation_eigenspace(jphase, basis=basis, num_spins=num_fermions * 4, shift=4,
                                  npmod=npmod)
