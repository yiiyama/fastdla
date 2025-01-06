"""Calculation of DLA."""
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import numpy as np
from numba import njit
from .sparse_pauli_vector import SparsePauliVector
from .spv_fast import _spv_commutator_fast, _spv_innerprod_fast, _spv_sum_fast


@njit
def _compute_xmatrix(basis_indices, basis_coeffs):
    size = len(basis_indices)
    xmat = np.eye(size, dtype=np.complex128) * 0.5
    for i1, (indices1, coeffs1) in enumerate(zip(basis_indices, basis_coeffs)):
        for i2, (indices2, coeffs2) in enumerate(zip(basis_indices[:i1], basis_coeffs[:i1])):
            xmat[i1, i2] = _spv_innerprod_fast(indices1, coeffs1, indices2, coeffs2)

    xmat += xmat.T.conjugate()
    return xmat


def compute_xmatrix(basis: Sequence[SparsePauliVector]) -> np.array:
    return _compute_xmatrix([op.indices for op in basis], [op.coeffs for op in basis])


@njit
def _is_independent(new_indices, new_coeffs, basis_indices, basis_coeffs, xmat_inv):
    basis_size = len(basis_indices)
    pidag_q = np.empty(basis_size, dtype=np.complex128)
    is_zero = True
    for ib, (bindex, bcoeff) in enumerate(zip(basis_indices, basis_coeffs)):
        ip = _spv_innerprod_fast(bindex, bcoeff, new_indices, new_coeffs)
        pidag_q[ib] = ip
        is_zero &= np.isclose(ip.real, 0.) and np.isclose(ip.imag, 0.)

    if is_zero:
        # Q is orthogonal to all basis vectors
        return True

    # Residual calculation: subtract Pi*ai from Q directly 
    for xinv_row, bindices, bcoeffs in zip(xmat_inv, basis_indices, basis_coeffs):
        a_val = xinv_row @ pidag_q
        if np.isclose(a_val.real, 0.) and np.isclose(a_val.imag, 0.):
            continue
        new_indices, new_coeffs = _spv_sum_fast(new_indices, new_coeffs,
                                                bindices, -a_val * bcoeffs)

    return new_indices.shape[0] != 0


def is_independent(
    new_op: SparsePauliVector,
    basis: list[SparsePauliVector],
    xmat_inv: np.ndarray,
) -> bool:
    """
    Let the known dla basis ops be P0, P1, ..., Pn. The basis_matrix Π is a matrix formed by
    stacking the column vectors {Pi}:
    Π = (P0, P1, ..., Pn).
    If new_op Q is linearly dependent on {Pi}, there is a column vector of coefficients
    a = (a0, a1, ..., an)^T where
    Π a = Q.
    Multiply both sides with Π† and denote X = Π†Π to obtain
    X a = Π† Q.
    Since {Pi} are linearly independent, X must be invertible:
    a = X^{-1} Π† Q.
    Using thus calculated {ai}, we check the residual
    R = Q - Π a
    to determine the linear independence of Q with respect to {Pi}.
    """
    return _is_independent(new_op.indices, new_op.coeffs,
                           [op.indices for op in basis], [op.coeffs for op in basis],
                           xmat_inv)


def full_dla_basis(generators: Sequence[SparsePauliVector]) -> list[SparsePauliVector]:
    basis_indices = [op.indices for op in generators]
    basis_coeffs = [op.coeffs / np.sqrt(np.sum(np.square(np.abs(op.coeffs)))) for op in generators]
    num_qubits = generators[0].num_qubits

    xmat_inv = np.linalg.inv(_compute_xmatrix(basis_indices, basis_coeffs))

    with ThreadPoolExecutor() as executor:
        commutators = [
            executor.submit(_spv_commutator_fast, idx1, c1, idx2, c2, num_qubits)
            for i1, (idx1, c1) in enumerate(zip(basis_indices, basis_coeffs))
            for idx2, c2 in zip(basis_indices[:i1], basis_coeffs[:i1])
        ]

        while commutators:
            done, _ = wait(commutators, return_when=FIRST_COMPLETED)
            for fut in done:
                commutators.remove(fut)
                indices, coeffs = fut.result()
                if indices.shape[0] == 0:
                    continue

                coeffs /= np.sqrt(np.sum(np.square(np.abs(coeffs))))
                if not _is_independent(indices, coeffs, basis_indices, basis_coeffs, xmat_inv):
                    continue

                commutators.extend([
                    executor.submit(_spv_commutator_fast, indices, coeffs, idx, c, num_qubits)
                    for idx, c in zip(basis_indices, basis_coeffs)
                ])

                basis_indices.append(indices)
                basis_coeffs.append(coeffs)
                xmat_inv = np.linalg.inv(_compute_xmatrix(basis_indices, basis_coeffs))

    return [SparsePauliVector(idx, c, num_qubits=num_qubits, no_check=True)
            for idx, c in zip(basis_indices, basis_coeffs)]
