"""Generators and symmetry projectors for Z2 lattice gauge theory HVA."""
from collections.abc import Sequence
from typing import Optional
import numpy as np
from ..sparse_pauli_vector import PAULIS, SparsePauliVector


def z2lgt_hva_generators(num_fermions: int) -> list[SparsePauliVector]:
    """Construct generators of the HVA for the Z2 LGT model with periodic boundary condition for the
    given number of staggered fermions."""
    num_qubits = 4 * num_fermions

    generators = []

    # Mass terms Z_{n}
    for parity in [0, 1]:
        strings = ['I' * (num_qubits - isite * 2 - 1) + 'Z' + 'I' * (isite * 2)
                   for isite in range(parity, 2 * num_fermions, 2)]
        coeffs = np.ones(len(strings)) / np.sqrt(len(strings))
        generators.append(SparsePauliVector(strings, coeffs))

    # Field term X_{n,n+1}
    strings = ['I' * (num_qubits - iq - 1) + 'X' + 'I' * iq for iq in range(1, num_qubits, 2)]
    coeffs = np.ones(len(strings)) / np.sqrt(len(strings))
    generators.append(SparsePauliVector(strings, coeffs))

    # Hopping terms X_{n}Z_{n,n+1}X_{n+1} + Y_{n}Z_{n,n+1}Y_{n+1}
    for parity in [0, 1]:
        strings = []
        for isite in range(parity, 2 * num_fermions, 2):
            for site_op in ['X', 'Y']:
                paulis_reverse = ['I'] * num_qubits
                paulis_reverse[isite * 2] = site_op
                paulis_reverse[isite * 2 + 1] = 'Z'
                paulis_reverse[(isite * 2 + 2) % num_qubits] = site_op
                strings.append(''.join(paulis_reverse[::-1]))
        coeffs = np.ones(len(strings)) / np.sqrt(len(strings))
        generators.append(SparsePauliVector(strings, coeffs))

    return generators


def z2lgt_gauss_local_projector(
    num_sites: int,
    isite: int,
    eigval: int
) -> SparsePauliVector:
    """Construct the Gauss's law projector for the Z2 LGT model for a single matter site.

    Physical states of the Z2 LGT model must be eigenstates of G_n = X_{n-1,n}Z_{n}X_{n1,n+1}. Given
    an eigenvalue for each G_n, we can construct a projector to its subspace as a sum of Pauli ops.
    """
    if abs(eigval) != 1:
        raise ValueError('Charge value must be +-1')

    num_qubits = 2 * num_sites

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
    paulis_reverse[(isite * 2 - 1) % num_qubits] = 'X'
    paulis_reverse[(isite * 2 + 1) % num_qubits] = 'X'
    strings.append(''.join(paulis_reverse[::-1]))

    return SparsePauliVector(strings, coeffs)


def z2lgt_gauss_projector(
    eigvals: Sequence[int]
) -> SparsePauliVector:
    """Construct the Gauss's law projector for the Z2 LGT model.

    Physical states of the Z2 LGT model must be eigenstates of G_n = X_{n-1,n}Z_{n}X_{n1,n+1}. Given
    an eigenvalue for each G_n, we can construct a projector to its subspace as a sum of Pauli ops.
    The overall projector will then be a product of all such projectors.
    """
    if (num_sites := len(eigvals)) % 2 or not all(abs(ev) == 1 for ev in eigvals):
        raise ValueError('There must be an even number of charges with values +-1')

    num_qubits = 2 * num_sites

    projector = SparsePauliVector('I' * num_qubits, 1.)
    for isite, ev in enumerate(eigvals):
        projector = projector @ z2lgt_gauss_local_projector(num_sites, isite, ev)

    return projector


def z2lgt_u1_projector(num_fermions: int, charge: int) -> SparsePauliVector:
    """Construct the charge conservation law projector for the Z2 LGT model.

    Physical states of the Z2 LGT model must be eigenstates of Q = ∑_n Z_n where n runs over
    staggered fermions (even n: particle, odd n: antiparticle). For a fermion number F and an
    overall charge q ∈ [-2F,...,2F], we have a 2F C (F+q/2) -dimensional eigenspace.
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

    return SparsePauliVector(paulis, coeffs)


def z2lgt_translation_projector(num_fermions: int, iroot: int) -> SparsePauliVector:
    """Construct the translation projector for the Z2 LGT model.

    The Z2 LGT HVA generators commute with the translation operator T, which shifts state indices
    by 1 fermion unit (2 sites, 4 qubits). Therefore, if the initial state of a VQA is a translation
    eigenstate, the variational evolution of the state stays within the same eigenspace.
    Since it is quite nontrivial to directly express the translation operator eigenspace projectors
    in terms of Pauli product sums, we first construct the projectors as dense matrices, then
    compute the inner products with Paulis.
    """
    if iroot < 0 or iroot >= num_fermions:
        raise ValueError('Invalid iroot value')

    num_qubits = 4 * num_fermions

    # Construct the translation eigenstates from computational basis states
    eigenstates = np.eye(2 ** num_qubits, dtype=np.complex128) / num_fermions
    powers = 1 << np.arange(num_qubits)[::-1]
    indices = np.arange(2 ** num_qubits)
    indices_bin = (indices[:, None] // powers[None, :]) % 2
    for shift in range(1, num_fermions):
        shifted_indices = np.sum(np.roll(indices_bin, 4 * shift, axis=1) * powers[None, :], axis=1)
        phase_factor = np.exp(-2.j * np.pi * iroot * shift / num_fermions)
        eigenstates[shifted_indices, indices] += phase_factor / num_fermions
    # Eigenstates = (|u_0>, |u_1>, ...)
    # Projector = sum_j |u_j><u_j|
    projector = eigenstates @ eigenstates.conjugate().T

    # Pauli decomposition through iterative partial traces
    projector = projector.reshape((2,) * (2 * num_qubits))
    for iq in range(num_qubits):
        projector = np.tensordot(PAULIS, projector, [[1, 2], [num_qubits, iq]]) / 2.

    # Sparsify
    projector = np.reshape(np.transpose(projector), -1)
    paulis = np.nonzero(projector)[0]
    coeffs = projector[paulis]

    return SparsePauliVector(paulis, coeffs)


def z2lgt_dense_projector(
    gauss_eigvals: Sequence[int],
    charge: Optional[int] = None,
    c_eigval: Optional[int] = None,
    t_iroot: Optional[int] = None,
    npmod=np
):
    """Construct a full symmetry projector for the Z2 LGT model.

    The construction follows the projection algorithm for multiple symmetries proposed by LN. See
    the docstring of individual projectors for details.

    Returns a (pu, 2**nq) matrix.
    """
    projector = z2lgt_dense_gauss_projector(gauss_eigvals, npmod=npmod)
    if charge is not None:
        projector = z2lgt_dense_u1_projector(projector, charge, npmod=npmod)
    if c_eigval is not None:
        projector = z2lgt_dense_charge_projector(projector, c_eigval, npmod=npmod)
    if t_iroot is not None:
        projector = z2lgt_dense_translation_projector(projector, t_iroot, npmod=npmod)

    return np.asarray(projector)


def z2lgt_dense_gauss_projector(
    gauss_eigvals: Sequence[int],
    npmod=np
) -> np.ndarray:
    """Construct the Gauss's law projector.

    The construction follows the projection algorithm for multiple symmetries proposed by LN. To
    avoid constructing the full 2^nq x 2^nq matrix, we compose the final projector from local-term
    projectors, extending the dimension as necessary.

    Algorithm:
    Let the number of matter sites be M, qubits be counted from right to left, and qubit 0
    correspond to matter site 0.
    1) Start with a local projector Q(0) for the left-most (M-1) XZX symmetry generator. Obtain a
        projector P(0)=Q(0) with shape (p(0), d(0)=8).
    2) For a quark site M-i-1 (i=1,...,M-2), extend the dimensions of the current projector P(i-1)
        by 4 to the right, then project the local Q(i) to
        R(i)=[P(i-1)⊗I⊗I][I⊗..⊗I⊗Q(i)][P(i-1)†⊗I⊗I] (shape (p(i-1)x4, p(i-1)x4)). Assemble the
        eigenvectors of R(i) with eigenvalue 1 into S(i)=(v_0..v_{p(i)}) (shape (p(i-1)x4, p(i)))
        and apply S(i)† to the current projector to obtain P(i)=S(i)†[P(i-1)⊗I⊗I]
        (shape (p(i), d(i)=d(i-1)x4)).
    3) For quark site 0 (i=M-1), extend the dimension of P(M-2) only by 2. The "local" projector
        actually acts on the leftmost qubit as well as the rightmost two. R(M-1) and S(M-1) have
        shapes (p(M-2)x2, p(M-2)x2) and (p(M-2)x2, p(M-1)) and the final projector will be
        (p(M-1), d(M-1)=d(M-2)x2=2**nq).
    """
    if len(gauss_eigvals) % 2 or not all(abs(ev) == 1 for ev in gauss_eigvals):
        raise ValueError('There must be an even number of charges with values +-1')

    num_sites = len(gauss_eigvals)
    projector = npmod.array(0.)

    # Gauss's law projector
    # Start from the leftmost link-site-link and iteratively construct the full-size projector
    for isite_r, ev in enumerate(gauss_eigvals[::-1]):
        if ev > 0:
            coeffs = [0.5, 0.5]
        else:
            coeffs = [0.5, -0.5]

        local_projector = SparsePauliVector(['III', 'XZX'], coeffs).to_matrix()
        if isite_r == 0:
            eigvals, eigvecs = npmod.linalg.eigh(local_projector)
            indices = npmod.nonzero(npmod.isclose(eigvals, 1.))[0]
            projector = eigvecs[:, indices].T.conjugate()
        elif isite_r == num_sites - 1:
            # Transpose the rightmost X to position 0 simultaneously with matrix multiplication
            pdim = projector.shape[0]
            projector = projector.reshape((pdim, 2, 2 ** (2 * num_sites - 3), 2))
            local_projector = local_projector.reshape((2, 2, 2, 2, 2, 2))
            projected_local = npmod.einsum('ijkl,lmjnop,qpkn->imqo', projector, local_projector,
                                           projector.conjugate())  # (pdim, 2, pdim, 2)
            projected_local = projected_local.reshape((pdim * 2,) * 2)
            eigvals, eigvecs = npmod.linalg.eigh(projected_local)
            indices = npmod.nonzero(npmod.isclose(eigvals, 1.))[0]
            subspace = eigvecs[:, indices].T.conjugate().reshape(-1, pdim, 2)
            projector = projector.reshape(pdim, 2 ** (2 * num_sites - 1))
            projector = npmod.einsum('ijk,jl->ilk', subspace, projector)
            projector = projector.reshape(-1, 2 ** (2 * num_sites))
        else:
            pdim = projector.shape[0]
            projector = projector.reshape((pdim, 2 ** (2 * isite_r), 2))
            local_projector = local_projector.reshape((2, 4, 2, 4))
            projected_local = npmod.einsum('ijk,klmn,ojm->ilon', projector, local_projector,
                                           projector.conjugate())  # (pdim, 4, pdim, 4)
            projected_local = projected_local.reshape((pdim * 4,) * 2)
            eigvals, eigvecs = npmod.linalg.eigh(projected_local)
            indices = npmod.nonzero(npmod.isclose(eigvals, 1.))[0]
            subspace = eigvecs[:, indices].T.conjugate().reshape(-1, pdim, 4)
            projector = projector.reshape(pdim, 2 ** (2 * isite_r + 1))
            projector = npmod.einsum('ijk,jl->ilk', subspace, projector)
            projector = projector.reshape(-1, 2 ** (2 * isite_r + 3))

    return projector


def z2lgt_dense_u1_projector(
    projector: np.ndarray,
    charge: int,
    npmod=np
) -> np.ndarray:
    """Project out the subspace corresponding to the U1 charge within the given projector.

    Because the U(1) projector is diagonal, we can apply a similar algorithm to combine the Gauss
    and U(1) projectors after the full Gauss projector is constructed. With the charge projector
    Qu (shape (2**nq, 2**nq) diagonal), first project it onto the Gauss subspace using P(M-1) to
    obtain Ru=P(M-1) Qu P(M-1)†. Then assemble the non-null eigenvectors into Su (shape (p(i), pu))
    and update the global projector to Pg=Su†P(M-1).
    """
    num_qubits = np.round(np.log2(projector.shape[1])).astype(int)
    num_sites = num_qubits // 2

    if abs(charge) > num_sites or charge % 2 != 0:
        raise ValueError('Invalid charge value')

    # Charge projection
    eigvals = npmod.zeros((2,) * num_sites, dtype=int)
    z = npmod.array([1, -1])
    for isite in range(num_sites):
        eigvals += npmod.expand_dims(z, tuple(range(isite)) + tuple(range(isite + 1, num_sites)))
    eigvals = eigvals.reshape(-1)
    states = npmod.nonzero(eigvals == charge)[0]

    # Project out the columns corresponding to wrong charge
    pdim = projector.shape[0]
    projector_reduced = projector.reshape(
        (pdim,) + (2,) * num_qubits
    ).transpose(
        (0,)
        + tuple(range(1, num_qubits + 1, 2))
        + tuple(range(2, num_qubits + 1, 2))
    ).reshape(
        (pdim,) + (2,) * num_sites + (2 ** num_sites,)
    )[..., states].reshape((pdim, -1))
    projected_charge = projector_reduced @ projector_reduced.conjugate().T

    eigvals, eigvecs = npmod.linalg.eigh(projected_charge)
    indices = npmod.nonzero(npmod.isclose(eigvals, 1.))[0]
    subspace = eigvecs[:, indices].T.conjugate()
    projector = subspace @ projector

    return projector


def z2lgt_dense_charge_projector(
    projector: np.ndarray,
    c_eigval: int,
    npmod=np
) -> np.ndarray:
    """Project out the subspace spanned by states with specific C eigenvalues from the projector.

    Charge conjugation is defined as a reflection about sites 0-1 followed by X on all sites.

    Since C does not commute with G_n and Q, we simply rely on linear algebra. Let P† represent
    the given projector array (shape (p, 2**nq)). A vector in the space spanned by columns of P is
    given by Pα, where α is a column vector with p entries. When this vector is an eigenvector of
    C,
        C Pα = c Pα ⇒ (C - cI)Pα = 0
    If the SVD of (C - cI)P is UΣV†, α is the conjugate of a row of V† corresponding to a zero
    singular value. We identify such αs and return the reduced projector [P (α0 α1 ...)]†.
    """
    if c_eigval not in [1, -1]:
        raise ValueError('Invalid C eigenvalue')

    num_qubits = np.round(np.log2(projector.shape[1])).astype(int)
    num_sites = num_qubits // 2

    pmat = projector.conjugate().T

    pdim = projector.shape[0]
    conjugate = pmat.reshape((2,) * num_qubits + (pdim,))
    swap = npmod.array([[1., 0., 0., 0.], [0., 0., 1., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.]])
    swap = swap.reshape((2,) * 4)
    paulix = npmod.array([[0., 1.], [1., 0.]])
    for iqubit in range(2, num_sites + 1):
        # Swap qubit i and 2 - i
        iax_src = num_qubits - iqubit - 1
        iax_dest = num_qubits - ((2 - iqubit) % num_qubits) - 1
        conjugate = npmod.moveaxis(
            npmod.tensordot(swap, conjugate, [[2, 3], [iax_src, iax_dest]]),
            [0, 1], [iax_src, iax_dest]
        )

    # Apply the X gate to each site
    for isite in range(num_sites):
        iax = num_qubits - isite * 2 - 1
        conjugate = npmod.moveaxis(
            npmod.tensordot(paulix, conjugate, [[1], [iax]]),
            0, iax
        )

    conjugate = conjugate.reshape((2 ** num_qubits, pdim))
    _, svals, vhmat = npmod.linalg.svd(conjugate - c_eigval * pmat, full_matrices=False)
    indices = npmod.nonzero(npmod.isclose(svals, 0.))[0]
    subspace = vhmat[indices]

    projector = subspace @ projector

    return projector


def z2lgt_dense_translation_projector(
    projector: np.ndarray,
    t_iroot: int,
    npmod=np
) -> np.ndarray:
    """Project out the subspace spanned by states with specific T eigenvalues from the projector.

    Translation is defined by a positive shift of 2 site units (4 qubits) on all qubits.

    Since T does not commute with G_n, we simply rely on linear algebra. Let P† represent
    the given projector array (shape (p, 2**nq)). A vector in the space spanned by columns of P is
    given by Pα, where α is a column vector with p entries. When this vector is an eigenvector of
    T,
        T Pα = ω^t Pα ⇒ (T - ω^tI)Pα = 0
    where ω is the Nf-th root of identity.
    If the SVD of (T - ω^tI)P is UΣV†, α is the conjugate of a row of V† corresponding to a zero
    singular value. We identify such αs and return the reduced projector [P (α0 α1 ...)]†.
    """
    num_qubits = np.round(np.log2(projector.shape[1])).astype(int)
    num_sites = num_qubits // 2
    num_fermions = num_sites // 2

    if t_iroot not in list(range(num_fermions)):
        raise ValueError('Invalid t_iroot value')

    pmat = projector.conjugate().T

    pdim = projector.shape[0]
    translated = pmat.reshape((2,) * num_qubits + (pdim,))
    swap = npmod.array([[1., 0., 0., 0.], [0., 0., 1., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.]])
    swap = swap.reshape((2,) * 4)
    for iax in range(num_qubits - 1, 3, -1):
        # Swap qubits i and i - 4
        translated = npmod.moveaxis(
            npmod.tensordot(swap, translated, [[2, 3], [iax, iax - 4]]),
            [0, 1], [iax, iax - 4]
        )

    translated = translated.reshape((2 ** num_qubits, pdim))
    eigval = np.exp(2.j * np.pi / num_fermions * t_iroot)
    _, svals, vhmat = np.linalg.svd(translated - eigval * pmat)
    indices = np.nonzero(np.isclose(svals, 0.))[0]
    subspace = vhmat[indices]

    projector = subspace @ projector

    return projector
