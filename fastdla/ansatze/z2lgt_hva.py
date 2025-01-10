"""Generators and symmetry projectors for Z2 lattice gauge theory HVA."""
from collections.abc import Sequence
import numpy as np
from ..sparse_pauli_vector import SparsePauliVector


def z2lgt_hva_generators(num_fermions: int) -> list[SparsePauliVector]:
    """Construct generators of the HVA for the Z2 LGT model for the given number of staggered
    fermions."""
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


def z2lgt_gauss_projector(
    eigvals: Sequence[int]
) -> SparsePauliVector:
    """Construct the Gauss's law projector for the Z2 LGT model.

    Physical states of the Z2 LGT model must be eigenstates of G_n = X_{n-1,n}Z_{n}X_{n1,n+1}. Given
    an eigenvalue for each G_n, we can construct projector to its subspace as a sum of Pauli ops.
    The overall projector will then be a product of all such projectors.

    If dense=True, returns a Sx2^{nq} matrix where S is the dimension of the projected space.
    """
    if len(eigvals) % 2 or not all(abs(ev) == 1 for ev in eigvals):
        raise ValueError('There must be an even number of charges with values +-1')

    num_fermions = len(eigvals) // 2
    num_qubits = 4 * num_fermions

    projector = SparsePauliVector('I' * num_qubits, 1.)
    for isite, ev in enumerate(eigvals):
        if ev > 0:
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
        projector = projector @ SparsePauliVector(strings, coeffs)

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


def z2lgt_dense_projector(gauss_eigvals: Sequence[int], charge: int, npmod=np):
    """Construct a full symmetry projector for the Z2 LGT model.

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

    Because the U(1) projector is diagonal, we can apply a similar algorithm to combine the Gauss
    and U(1) projectors after the full Gauss projector is constructed. With the charge projector
    Qu (shape (2**nq, 2**nq) diagonal), first project it onto the Gauss subspace using P(M-1) to
    obtain Ru=P(M-1) Qu P(M-1)†. Then assemble the non-null eigenvectors into Su (shape (p(i), pu))
    and update the global projector to Pg=Su†P(M-1).

    Returns a (pu, 2**nq) matrix.
    """
    if len(gauss_eigvals) % 2 or not all(abs(ev) == 1 for ev in gauss_eigvals):
        raise ValueError('There must be an even number of charges with values +-1')

    num_sites = len(gauss_eigvals)
    num_qubits = 2 * num_sites

    if abs(charge) > num_sites or charge % 2 != 0:
        raise ValueError('Invalid charge value')

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

    return np.asarray(projector)
