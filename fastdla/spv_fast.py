"""Numba-compiled versions of SparsePauliVector operations."""
import numpy as np
from numba import njit
from .sparse_pauli_vector import PAULI_PROD_COEFF, PAULI_PROD_INDEX, SparsePauliVector


@njit(nogil=True)
def _uniquify_fast(indices: np.ndarray, coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sidx = np.argsort(indices)
    indices_sorted = indices[sidx]
    coeffs_sorted = coeffs[sidx]

    indices_unique = np.empty_like(indices)
    coeffs_unique = np.empty_like(coeffs)
    indices_unique[0] = indices_sorted[0]
    coeffs_unique[0] = coeffs_sorted[0]

    iout = 0
    for index, coeff in zip(indices_sorted[1:], coeffs_sorted[1:]):
        if index == indices_unique[iout]:
            coeffs_unique[iout] += coeff
        else:
            if not (np.isclose(coeffs_unique[iout].real, 0.)
                    and np.isclose(coeffs_unique[iout].imag, 0.)):
                iout += 1
            indices_unique[iout] = index
            coeffs_unique[iout] = coeff

    if not (np.isclose(coeffs_unique[iout].real, 0.)
            and np.isclose(coeffs_unique[iout].imag, 0.)):
        iout += 1

    return indices_unique[:iout], coeffs_unique[:iout]


@njit(nogil=True)
def _spv_prod_fast(
    indices1: np.ndarray,
    coeffs1: np.ndarray,
    indices2: np.ndarray,
    coeffs2: np.ndarray,
    num_qubits: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compiled version of SparsePauliVector product."""
    coeffs = np.outer(coeffs1, coeffs2)
    indices = np.zeros((indices1.shape[0], indices2.shape[0]), dtype=indices1.dtype)
    for iq in range(num_qubits):
        shift = 4 ** iq
        paulis1 = np.asarray((indices1 // shift) % 4, dtype=np.int32)
        paulis2 = np.asarray((indices2 // shift) % 4, dtype=np.int32)
        for i1, p1 in enumerate(paulis1):
            for i2, p2 in enumerate(paulis2):
                indices[i1, i2] += PAULI_PROD_INDEX[p1, p2] * shift
                coeffs[i1, i2] *= PAULI_PROD_COEFF[p1, p2]

    indices = indices.reshape(-1)
    coeffs = coeffs.reshape(-1)
    return _uniquify_fast(indices, coeffs)


def spv_prod_fast(lhs: SparsePauliVector, rhs: SparsePauliVector) -> SparsePauliVector:
    if lhs.num_qubits != lhs.num_qubits:
        raise ValueError('Matmul between incompatible SparsePauliVectors')
    if lhs.num_terms * lhs.num_terms == 0:
        return SparsePauliVector([], [], lhs.num_qubits, no_check=True)

    indices, coeffs = _spv_prod_fast(lhs.indices, lhs.coeffs, rhs.indices, rhs.coeffs,
                                     lhs.num_qubits)
    return SparsePauliVector(indices, coeffs, lhs.num_qubits, no_check=True)


@njit(nogil=True)
def _spv_commutator_fast(
    indices1: np.ndarray,
    coeffs1: np.ndarray,
    indices2: np.ndarray,
    coeffs2: np.ndarray,
    num_qubits: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compiled version of SparsePauliVector commutator."""
    coeffs = np.zeros((indices1.shape[0], indices2.shape[0]), dtype=coeffs1.dtype)
    indices = np.zeros((indices1.shape[0], indices2.shape[0]), dtype=indices1.dtype)
    iqs = np.arange(num_qubits)
    shifts = 4 ** iqs
    paulis1 = np.asarray((indices1[:, None] // shifts[None, :]) % 4, dtype=np.int32)
    paulis2 = np.asarray((indices2[:, None] // shifts[None, :]) % 4, dtype=np.int32)
    for i1, (pp1, coeff1) in enumerate(zip(paulis1, coeffs1)):
        for i2, (pp2, coeff2) in enumerate(zip(paulis2, coeffs2)):
            pauli_prod_coeff = 1.
            for iq in iqs:
                pauli_prod_coeff *= PAULI_PROD_COEFF[pp1[iq], pp2[iq]]
                indices[i1, i2] += PAULI_PROD_INDEX[pp1[iq], pp2[iq]] * shifts[iq]

            if np.isclose(pauli_prod_coeff.imag, 0.):
                continue

            coeffs[i1, i2] = 2.j * coeff1 * coeff2 * pauli_prod_coeff.imag

    indices = indices.reshape(-1)
    coeffs = coeffs.reshape(-1)
    return _uniquify_fast(indices, coeffs)


def spv_commutator_fast(lhs: SparsePauliVector, rhs: SparsePauliVector) -> SparsePauliVector:
    if lhs.num_qubits != lhs.num_qubits:
        raise ValueError('Matmul between incompatible SparsePauliVectors')
    if lhs.num_terms * lhs.num_terms == 0:
        return SparsePauliVector([], [], lhs.num_qubits, no_check=True)

    indices, coeffs = _spv_commutator_fast(lhs.indices, lhs.coeffs, rhs.indices, rhs.coeffs,
                                           lhs.num_qubits)
    return SparsePauliVector(indices, coeffs, lhs.num_qubits, no_check=True)
