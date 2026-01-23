"""Numba-compiled versions of SparsePauliSum operations."""
import logging
from typing import Optional
import numpy as np
from numba import njit
from .pauli import PAULI_MULT_COEFF, PAULI_MULT_INDEX
from .sparse_pauli_sum import SparsePauliSum, SparsePauliSumArray
from .linalg.numba_ops import complex_isclose, _check_finite

LOG = logging.getLogger(__name__)
MEM_ALLOC_UNIT = SparsePauliSumArray.MEM_ALLOC_UNIT


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

    normsq = 0.
    iout = 0
    atol_real = abs(coeffs_unique[0].real)
    atol_imag = abs(coeffs_unique[0].imag)
    for index, coeff in zip(indices_sorted[1:], coeffs_sorted[1:]):
        if index == indices_unique[iout]:
            coeffs_unique[iout] += coeff
            atol_real += abs(coeff.real)
            atol_imag += abs(coeff.imag)
        else:
            iout, normsq = _check_finite(coeffs_unique[iout], atol_real, atol_imag, iout, normsq)
            indices_unique[iout] = index
            coeffs_unique[iout] = coeff
            atol_real = abs(coeff.real)
            atol_imag = abs(coeff.imag)

    iout, normsq = _check_finite(coeffs_unique[iout], atol_real, atol_imag, iout, normsq)

    if normalize and iout > 0:
        coeffs_unique[:iout] /= np.sqrt(normsq)

    return indices_unique[:iout], coeffs_unique[:iout]


@njit(nogil=True)
def _sps_add_fast(
    indices1: np.ndarray,
    coeffs1: np.ndarray,
    indices2: np.ndarray,
    coeffs2: np.ndarray,
    normalize: bool
) -> tuple[np.ndarray, np.ndarray]:
    return _uniquify_fast(
        np.concatenate((indices1, indices2)),
        np.concatenate((coeffs1, coeffs2)),
        normalize
    )


def sps_add_fast(
    lhs: SparsePauliSum,
    rhs: SparsePauliSum,
    normalize: bool = False
) -> SparsePauliSum:
    if lhs.num_qubits != lhs.num_qubits:
        raise ValueError('Sum between incompatible SparsePauliSums')
    if lhs.num_terms * lhs.num_terms == 0:
        return SparsePauliSum([], [], num_qubits=lhs.num_qubits, no_check=True)

    indices, coeffs = _sps_add_fast(lhs.indices, lhs.coeffs, rhs.indices, rhs.coeffs, normalize)
    return SparsePauliSum(indices, coeffs, num_qubits=lhs.num_qubits, no_check=True)


@njit(nogil=True)
def _sps_matmul_fast(
    indices1: np.ndarray,
    coeffs1: np.ndarray,
    indices2: np.ndarray,
    coeffs2: np.ndarray,
    num_qubits: int,
    normalize: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Compiled version of SparsePauliSum product."""
    coeffs = np.outer(coeffs1, coeffs2)
    indices = np.zeros((indices1.shape[0], indices2.shape[0]), dtype=indices1.dtype)
    for iq in range(num_qubits):
        shift = 4 ** iq
        paulis1 = np.asarray((indices1 // shift) % 4, dtype=np.int32)
        paulis2 = np.asarray((indices2 // shift) % 4, dtype=np.int32)
        for i1, p1 in enumerate(paulis1):
            for i2, p2 in enumerate(paulis2):
                indices[i1, i2] += PAULI_MULT_INDEX[p1, p2] * shift
                coeffs[i1, i2] *= PAULI_MULT_COEFF[p1, p2]

    indices = indices.reshape(-1)
    coeffs = coeffs.reshape(-1)
    return _uniquify_fast(indices, coeffs, normalize)


def sps_matmul_fast(
    lhs: SparsePauliSum,
    rhs: SparsePauliSum,
    normalize: bool = False
) -> SparsePauliSum:
    if lhs.num_qubits != lhs.num_qubits:
        raise ValueError('Matmul between incompatible SparsePauliSums')
    if lhs.num_terms * lhs.num_terms == 0:
        return SparsePauliSum([], [], num_qubits=lhs.num_qubits, no_check=True)

    indices, coeffs = _sps_matmul_fast(lhs.indices, lhs.coeffs, rhs.indices, rhs.coeffs,
                                       lhs.num_qubits, normalize)
    return SparsePauliSum(indices, coeffs, num_qubits=lhs.num_qubits, no_check=True)


@njit(nogil=True)
def _sps_commutator_fast(
    indices1: np.ndarray,
    coeffs1: np.ndarray,
    indices2: np.ndarray,
    coeffs2: np.ndarray,
    num_qubits: int,
    normalize: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Compiled version of SparsePauliSum commutator."""
    coeffs = np.zeros((indices1.shape[0], indices2.shape[0]), dtype=coeffs1.dtype)
    indices = np.zeros((indices1.shape[0], indices2.shape[0]), dtype=indices1.dtype)
    iqs = np.arange(num_qubits)
    shifts = 4 ** iqs
    paulis1 = np.asarray((indices1[:, None] // shifts[None, :]) % 4, dtype=np.int32)
    paulis2 = np.asarray((indices2[:, None] // shifts[None, :]) % 4, dtype=np.int32)
    for i1, (pp1, coeff1) in enumerate(zip(paulis1, coeffs1)):
        for i2, (pp2, coeff2) in enumerate(zip(paulis2, coeffs2)):
            pauli_mult_coeff = 1.
            for iq in iqs:
                pauli_mult_coeff *= PAULI_MULT_COEFF[pp1[iq], pp2[iq]]
                indices[i1, i2] += PAULI_MULT_INDEX[pp1[iq], pp2[iq]] * shifts[iq]

            if np.isclose(pauli_mult_coeff.imag, 0.):
                continue

            coeffs[i1, i2] = 2.j * coeff1 * coeff2 * pauli_mult_coeff.imag

    indices = indices.reshape(-1)
    coeffs = coeffs.reshape(-1)
    return _uniquify_fast(indices, coeffs, normalize)


def sps_commutator_fast(
    lhs: SparsePauliSum,
    rhs: SparsePauliSum,
    normalize: bool = False
) -> SparsePauliSum:
    if lhs.num_qubits != lhs.num_qubits:
        raise ValueError('Commutator between incompatible SparsePauliSums')
    if lhs.num_terms * lhs.num_terms == 0:
        return SparsePauliSum([], [], num_qubits=lhs.num_qubits, no_check=True)

    indices, coeffs = _sps_commutator_fast(lhs.indices, lhs.coeffs, rhs.indices, rhs.coeffs,
                                           lhs.num_qubits, normalize)
    return SparsePauliSum(indices, coeffs, num_qubits=lhs.num_qubits, no_check=True)


def _sps_dot_fast_impl(
    indices1: np.ndarray,
    coeffs1: np.ndarray,
    indices2: np.ndarray,
    coeffs2: np.ndarray
) -> complex:
    i1 = 0
    i2 = 0
    result = 0.+0.j
    atol_real = 0.
    atol_imag = 0.
    while i1 < indices1.shape[0] and i2 < indices2.shape[0]:
        if indices1[i1] == indices2[i2]:
            prod = np.conjugate(coeffs1[i1]) * coeffs2[i2]
            result += prod
            atol_real += abs(prod.real)
            atol_imag += abs(prod.imag)
            i1 += 1
            i2 += 1
        elif indices1[i1] > indices2[i2]:
            i2 += 1
        else:
            i1 += 1

    atol = max(atol_real, atol_imag) * 1.e-5
    if complex_isclose(result, 0., atol=atol):
        return 0.+0.j
    return result


_sps_dot_fast = njit(nogil=True)(_sps_dot_fast_impl)
_sps_dot_fast_inline = njit(nogil=True, inline='always')(_sps_dot_fast_impl)


def sps_dot_fast(lhs: SparsePauliSum, rhs: SparsePauliSum) -> complex:
    if lhs.num_qubits != lhs.num_qubits:
        raise ValueError('Inner product between incompatible SparsePauliSums')
    if lhs.num_terms * lhs.num_terms == 0:
        return SparsePauliSum([], [], num_qubits=lhs.num_qubits, no_check=True)

    return _sps_dot_fast(lhs.indices, lhs.coeffs, rhs.indices, rhs.coeffs)


@njit(nogil=True)
def _spsarray_append_fast(
    array_indices: np.ndarray,
    array_coeffs: np.ndarray,
    array_ptrs: list[int],
    indices: np.ndarray,
    coeffs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    next_ptr = array_ptrs[-1] + indices.shape[0]
    if next_ptr > array_indices.shape[0]:
        # At maximum capacity -> reallocate
        additional_capacity = (((next_ptr - array_indices.shape[0]) // MEM_ALLOC_UNIT + 1)
                               * MEM_ALLOC_UNIT)
        array_indices = np.concatenate((
            array_indices,
            np.empty(additional_capacity, dtype=array_indices.dtype)
        ))
        array_coeffs = np.concatenate((
            array_coeffs,
            np.empty(additional_capacity, dtype=array_coeffs.dtype)
        ))

    array_ptrs.append(next_ptr)
    array_indices[array_ptrs[-2]:array_ptrs[-1]] = indices
    array_coeffs[array_ptrs[-2]:array_ptrs[-1]] = coeffs

    return array_indices, array_coeffs


def spsarray_append_fast(
    array: SparsePauliSumArray,
    arg1: SparsePauliSum | np.ndarray,
    arg2: Optional[np.ndarray] = None
):
    if arg2 is None:
        if array.num_qubits is None:
            array.num_qubits = arg1.num_qubits
        elif arg1.num_qubits != array.num_qubits:
            raise ValueError('Inconsistent num_qubits')
        indices = arg1.indices
        coeffs = arg1.coeffs
    else:
        indices = arg1
        coeffs = arg2

    array.indices, array.coeffs = _spsarray_append_fast(array.indices, array.coeffs, array.ptrs,
                                                        indices, coeffs)
