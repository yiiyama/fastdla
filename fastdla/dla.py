"""Calculation of DLA."""
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import numpy as np
from numba import njit
from .sparse_pauli_vector import SparsePauliVector
from .spv_fast import _spv_commutator_fast, _spv_innerprod_fast, _spv_sum_fast


@njit
def _fill_xmatrix_lower(indices_list, coeffs_list, xmat):
    for i1, (indices1, coeffs1) in enumerate(zip(indices_list, coeffs_list)):
        for i2, (indices2, coeffs2) in enumerate(zip(indices_list[:i1], coeffs_list[:i1])):
            xmat[i1, i2] = _spv_innerprod_fast(indices1, coeffs1, indices2, coeffs2)


def compute_xmatrix(basis: list[SparsePauliVector]) -> np.array:
    size = len(basis)
    xmat = np.eye(size, dtype=np.complex128) * 0.5
    _fill_xmatrix_lower([op.indices for op in basis], [op.coeffs for op in basis], xmat)
    xmat += xmat.T.conjugate()
    return xmat


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

    a_proj = xmat_inv @ pidag_q
    a_nonzero = np.nonzero(a_proj)[0]
    # a_proj cannot be a null vector because xmat_inv is nonsingular and pidag_q is nonnull
    pia_indices = basis_indices[a_nonzero[0]]
    pia_coeffs = basis_coeffs[a_nonzero[0]] * a_proj[a_nonzero[0]]
    for a_idx in a_nonzero[1:]:
        pia_indices, pia_coeffs = _spv_sum_fast(pia_indices, pia_coeffs,
                                                basis_indices[a_idx],
                                                basis_coeffs[a_idx] * a_proj[a_idx])

    res_indices, _ = _spv_sum_fast(new_indices, new_coeffs, pia_indices, -pia_coeffs)
    return res_indices.shape[0] != 0


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
    basis = [op.normalize() for op in generators]
    num_qubits = generators[0].num_qubits

    xmat_inv = np.linalg.inv(compute_xmatrix(basis))

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(_spv_commutator_fast,
                            op1.indices, op1.coeffs, op2.indices, op2.coeffs, num_qubits)
            for i1, op1 in enumerate(basis) for op2 in basis[:i1]
        ]

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.remove(future)
                indices, coeffs = future.result()
                if indices.shape[0] == 0:
                    continue

                coeffs /= np.sqrt(np.sum(np.square(np.abs(coeffs))))
                new_op = SparsePauliVector(indices, coeffs, num_qubits, no_check=True)
                if not is_independent(new_op, basis, xmat_inv):
                    continue

                futures.extend([
                    executor.submit(_spv_commutator_fast,
                                    new_op.indices, new_op.coeffs, op.indices, op.coeffs,
                                    num_qubits)
                    for op in basis
                ])
                basis.append(new_op)
                xmat_inv = np.linalg.inv(compute_xmatrix(basis))

    return basis
