"""Numba-compiled versions of SparsePauliVector operations."""
import numpy as np
from numba import njit
from .sparse_pauli_vector import PAULI_PROD_COEFF, PAULI_PROD_INDEX, SparsePauliVector


@njit(nogil=True)
def _uniquify_fast(
    indices: np.ndarray,
    coeffs: np.ndarray,
    normalize: bool
) -> tuple[np.ndarray, np.ndarray]:
    sidx = np.argsort(indices)
    indices_sorted = indices[sidx]
    coeffs_sorted = coeffs[sidx]

    indices_unique = np.empty_like(indices)
    coeffs_unique = np.empty_like(coeffs)
    indices_unique[0] = indices_sorted[0]
    coeffs_unique[0] = coeffs_sorted[0]
    normsq = np.square(coeffs_unique[0].real) + np.square(coeffs_unique[0].imag)

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
            normsq += np.square(coeff.real) + np.square(coeff.imag)

    if not (np.isclose(coeffs_unique[iout].real, 0.)
            and np.isclose(coeffs_unique[iout].imag, 0.)):
        iout += 1

    if normalize:
        coeffs_unique /= np.sqrt(normsq)

    return indices_unique[:iout], coeffs_unique[:iout]


@njit(nogil=True)
def _spv_sum_fast(
    indices1: np.ndarray,
    coeffs1: np.ndarray,
    indices2: np.ndarray,
    coeffs2: np.ndarray,
    normalize: bool
) -> tuple[np.ndarray, np.ndarray]:
    i1 = 0
    i2 = 0
    iout = 0
    indices = np.empty(indices1.shape[0] + indices2.shape[0], dtype=indices1.dtype)
    coeffs = np.empty(indices1.shape[0] + indices2.shape[0], dtype=coeffs1.dtype)
    normsq = 0.
    while i1 < indices1.shape[0] and i2 < indices2.shape[0]:
        if indices1[i1] == indices2[i2]:
            index = indices1[i1]
            coeff = coeffs1[i1] + coeffs2[i2]
            i1 += 1
            i2 += 1
            if not (np.isclose(coeff.real, 0.) and np.isclose(coeff.imag, 0.)):
                indices[iout] = index
                coeffs[iout] = coeff
                normsq += np.square(coeff.real) + np.square(coeff.imag)
                iout += 1
        elif indices1[i1] > indices2[i2]:
            indices[iout] = indices2[i2]
            coeffs[iout] = coeffs2[i2]
            normsq += np.square(coeffs2[i2].real) + np.square(coeffs2[i2].imag)
            i2 += 1
            iout += 1
        else:
            indices[iout] = indices1[i1]
            coeffs[iout] = coeffs1[i1]
            normsq += np.square(coeffs1[i1].real) + np.square(coeffs1[i1].imag)
            i1 += 1
            iout += 1

    r1 = indices1.shape[0] - i1
    r2 = indices2.shape[0] - i2
    if r1 > 0:
        indices[iout:iout + r1] = indices1[i1:]
        coeffs[iout:iout + r1] = coeffs1[i1:]
        normsq += np.sum(np.square(coeffs1[i1:].real) + np.square(coeffs1[i1:].imag))
        iout += r1
    elif r2 > 0:
        indices[iout:iout + r2] = indices2[i2:]
        coeffs[iout:iout + r2] = coeffs2[i2:]
        normsq += np.sum(np.square(coeffs2[i2:].real) + np.square(coeffs2[i2:].imag))
        iout += r2

    if normalize:
        coeffs /= np.sqrt(normsq)

    return indices[:iout], coeffs[:iout]


def spv_sum_fast(
    lhs: SparsePauliVector,
    rhs: SparsePauliVector,
    normalize: bool = False
) -> SparsePauliVector:
    if lhs.num_qubits != lhs.num_qubits:
        raise ValueError('Sum between incompatible SparsePauliVectors')
    if lhs.num_terms * lhs.num_terms == 0:
        return SparsePauliVector([], [], lhs.num_qubits, no_check=True)

    indices, coeffs = _spv_sum_fast(lhs.indices, lhs.coeffs, rhs.indices, rhs.coeffs, normalize)
    return SparsePauliVector(indices, coeffs, lhs.num_qubits, no_check=True)


@njit(nogil=True)
def _spv_prod_fast(
    indices1: np.ndarray,
    coeffs1: np.ndarray,
    indices2: np.ndarray,
    coeffs2: np.ndarray,
    num_qubits: int,
    normalize: bool
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
    return _uniquify_fast(indices, coeffs, normalize)


def spv_prod_fast(
    lhs: SparsePauliVector,
    rhs: SparsePauliVector,
    normalize: bool = False
) -> SparsePauliVector:
    if lhs.num_qubits != lhs.num_qubits:
        raise ValueError('Matmul between incompatible SparsePauliVectors')
    if lhs.num_terms * lhs.num_terms == 0:
        return SparsePauliVector([], [], lhs.num_qubits, no_check=True)

    indices, coeffs = _spv_prod_fast(lhs.indices, lhs.coeffs, rhs.indices, rhs.coeffs,
                                     lhs.num_qubits, normalize)
    return SparsePauliVector(indices, coeffs, lhs.num_qubits, no_check=True)


@njit(nogil=True)
def _spv_commutator_fast(
    indices1: np.ndarray,
    coeffs1: np.ndarray,
    indices2: np.ndarray,
    coeffs2: np.ndarray,
    num_qubits: int,
    normalize: bool
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
    return _uniquify_fast(indices, coeffs, normalize)


def spv_commutator_fast(
    lhs: SparsePauliVector,
    rhs: SparsePauliVector,
    normalize: bool = False
) -> SparsePauliVector:
    if lhs.num_qubits != lhs.num_qubits:
        raise ValueError('Matmul between incompatible SparsePauliVectors')
    if lhs.num_terms * lhs.num_terms == 0:
        return SparsePauliVector([], [], lhs.num_qubits, no_check=True)

    indices, coeffs = _spv_commutator_fast(lhs.indices, lhs.coeffs, rhs.indices, rhs.coeffs,
                                           lhs.num_qubits, normalize)
    return SparsePauliVector(indices, coeffs, lhs.num_qubits, no_check=True)


@njit(nogil=True)
def _spv_innerprod_fast(
    indices1: np.ndarray,
    coeffs1: np.ndarray,
    indices2: np.ndarray,
    coeffs2: np.ndarray
) -> complex:
    i1 = 0
    i2 = 0
    result = 0.+0.j
    while i1 < indices1.shape[0] and i2 < indices2.shape[0]:
        if indices1[i1] == indices2[i2]:
            result += np.conjugate(coeffs1[i1]) * coeffs2[i2]
            i1 += 1
            i2 += 1
        elif indices1[i1] > indices2[i2]:
            i2 += 1
        else:
            i1 += 1

    return result


def spv_innerprod_fast(lhs: SparsePauliVector, rhs: SparsePauliVector) -> complex:
    if lhs.num_qubits != lhs.num_qubits:
        raise ValueError('Matmul between incompatible SparsePauliVectors')
    if lhs.num_terms * lhs.num_terms == 0:
        return SparsePauliVector([], [], lhs.num_qubits, no_check=True)

    return _spv_innerprod_fast(lhs.indices, lhs.coeffs, rhs.indices, rhs.coeffs)
