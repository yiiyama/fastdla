"""Purpose-specific implementation of a sparse Pauli operator."""
from collections.abc import Sequence
from numbers import Number
from typing import Union
import numpy as np
from scipy.sparse import csc_array

PAULI_NAMES = ['I', 'X', 'Y', 'Z']
PAULI_INDICES = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}

PAULI_PROD_INDEX = np.array([
    [0, 1, 2, 3],
    [1, 0, 3, 2],
    [2, 3, 0, 1],
    [3, 2, 1, 0]
], dtype=np.uint64)
PAULI_PROD_COEFF = np.array([
    [1., 1., 1., 1.],
    [1., 1., 1.j, -1.j],
    [1., -1.j, 1., 1.j],
    [1., 1.j, -1.j, 1.]
], dtype=np.complex128)


class SparsePauliVector:
    """Purpose-specific implementation of a sparse Pauli operator."""
    @staticmethod
    def str_to_idx(pstr: str) -> int:
        return np.ravel_multi_index(tuple(PAULI_INDICES[p] for p in pstr),
                                    (4,) * len(pstr))

    @staticmethod
    def idx_to_str(idx: int, num_qubits: int) -> str:
        return ''.join(PAULI_NAMES[int(i)] for i in np.unravel_index(idx, (4,) * num_qubits))

    def __init__(
        self,
        strings: Union[str, int, Sequence[str], Sequence[int]],
        coeffs: Union[Number, Sequence[Number]],
        num_qubits=None,
        no_check=False
    ):
        if no_check:
            self.indices = np.asarray(strings, dtype=np.uint64)
            self.coeffs = np.asarray(coeffs, dtype=np.complex128)
            self.num_qubits = num_qubits
            return

        if isinstance(strings, (str, int)):
            strings = [strings]
        if isinstance(coeffs, Number):
            coeffs = [coeffs]

        if len(strings) == 0:
            self.num_qubits = num_qubits
            self.indices = np.array([], dtype=np.uint64)
            self.coeffs = np.array([], dtype=np.complex128)
            return

        if isinstance(strings[0], str):
            num_qubits = len(strings[0])
            indices = np.array([self.str_to_idx(pstr) for pstr in strings], dtype=np.uint64)
        elif num_qubits is None:
            raise ValueError('Need num_qubits')
        else:
            indices = np.asarray(strings, dtype=np.uint64)

        self.num_qubits = num_qubits
        sort_idx = np.argsort(indices)
        self.indices = indices[sort_idx]
        self.coeffs = np.asarray(coeffs, dtype=np.complex128)[sort_idx]

        if self.indices.shape != self.coeffs.shape:
            raise ValueError('Strings and coeffs shape mismatch')

    def __str__(self) -> str:
        opstr = ', '.join(f'{coeff} {self.idx_to_str(idx, self.num_qubits)}'
                          for coeff, idx in zip(self.coeffs, self.indices))
        return f'[{opstr}]'

    def __repr__(self) -> str:
        return ('SparsePauliVector(['
                + (', '.join(f"'{self.idx_to_str(idx, self.num_qubits)}'" for idx in self.indices))
                + '], coeffs=['
                + (', '.join(f'{coeff:.2f}' for coeff in self.coeffs))
                + '])')

    @property
    def shape(self) -> tuple[int]:
        return (4,) * self.num_qubits

    @property
    def vlen(self) -> int:
        return 4**self.num_qubits

    @property
    def num_terms(self) -> int:
        return self.indices.shape[0]

    @property
    def hpart(self) -> 'SparsePauliVector':
        """Return a new SPO with Hermitian terms only."""
        indices = np.nonzero(np.logical_not(np.isclose(self.coeffs.real, 0.)))[0]
        return SparsePauliVector(self.indices[indices], self.coeffs[indices].real, self.num_qubits)

    @property
    def apart(self) -> 'SparsePauliVector':
        """Return a new SPO with anti-Hermitian terms only."""
        indices = np.nonzero(np.logical_not(np.isclose(self.coeffs.imag, 0.)))[0]
        return SparsePauliVector(self.indices[indices], 1.j * self.coeffs[indices].imag,
                                 self.num_qubits)

    def normalize(self) -> 'SparsePauliVector':
        norm = np.sqrt(np.sum(np.square(np.abs(self.coeffs))))
        return SparsePauliVector(self.indices.copy(), self.coeffs / norm, self.num_qubits)

    def __add__(self, other: 'SparsePauliVector') -> 'SparsePauliVector':
        return spv_sum(self, other)

    def __mul__(self, scalar: Number) -> 'SparsePauliVector':
        try:
            coeffs = scalar * self.coeffs
        except Exception:  # pylint: disable=broad-exception-caught
            return NotImplemented
        return SparsePauliVector(self.indices, coeffs, self.num_qubits)

    def __rmul__(self, scalar: Number) -> 'SparsePauliVector':
        return self.__mul__(scalar)

    def __matmul__(self, other: 'SparsePauliVector') -> 'SparsePauliVector':
        return spv_prod(self, other)

    def commutator(self, other: 'SparsePauliVector') -> 'SparsePauliVector':
        return spv_commutator(self, other)

    def to_csc(self) -> csc_array:
        shape = (self.vlen, 1)
        return csc_array((self.coeffs, self.indices, [0, self.num_terms]), shape=shape)

    @staticmethod
    def from_csc(array: csc_array, num_qubits: int) -> 'SparsePauliVector':
        return SparsePauliVector(array.indices, array.data, num_qubits)


def _uniquify(strings: np.ndarray, coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Uniquify a pauli list by summing up the cofficients of overlapping operators."""
    strings_uniq, unique_indices = np.unique(strings, return_inverse=True)
    if strings_uniq.shape[0] == strings.shape[0]:
        # strings is already unique
        return strings_uniq, coeffs[np.argsort(unique_indices)]

    # strings is not unique -> merge duplicates
    coeffs_matrix = np.zeros((coeffs.shape[0], strings_uniq.shape[0]), dtype=coeffs.dtype)
    coeffs_matrix[np.arange(coeffs.shape[0]), unique_indices] = coeffs
    return strings_uniq, np.sum(coeffs_matrix, axis=0)


def spv_sum(*ops) -> SparsePauliVector:
    """Sum of two sparse Pauli ops."""
    strings, coeffs = _uniquify(
        np.concatenate([op.strings for op in ops]),
        np.concatenate([op.coeffs for op in ops])
    )
    return SparsePauliVector(strings, coeffs, ops[0].num_qubits)


def spv_prod(o1: SparsePauliVector, o2: SparsePauliVector) -> SparsePauliVector:
    """Product of two sparse Pauli ops."""
    if o1.num_qubits != o2.num_qubits:
        raise ValueError('Matmul between incompatible SparsePauliOps')

    if (num_terms := o1.num_terms * o2.num_terms) == 0:
        return SparsePauliVector([], [], o1.num_qubits)

    indices = np.unravel_index(np.arange(num_terms), (o1.num_terms, o2.num_terms))
    coeffs = np.outer(o1.coeffs, o2.coeffs).reshape(-1)
    s1s = o1.indices[indices[0]]
    s2s = o2.indices[indices[1]]

    shape = o1.shape
    p1s = np.array(np.unravel_index(s1s, shape=shape))  # shape [num_qubits, num_strings]
    p2s = np.array(np.unravel_index(s2s, shape=shape))
    pout = PAULI_PROD_INDEX[p1s, p2s]
    sout = np.ravel_multi_index(tuple(pout), shape)  # shape [num_strings]
    coeffs *= np.prod(PAULI_PROD_COEFF[p1s, p2s], axis=0)

    strings, coeffs = _uniquify(sout, coeffs)
    return SparsePauliVector(strings, coeffs, o1.num_qubits)


def spv_commutator(o1: SparsePauliVector, o2: SparsePauliVector) -> SparsePauliVector:
    o1h = o1.hpart
    o1a = o1.apart
    o2h = o2.hpart
    o2a = o2.apart
    comms = []
    for lhs, rhs in [(o1h, o2h), (o1a, o2a)]:
        if lhs.num_terms * rhs.num_terms != 0:
            comms.append(2. * spv_prod(lhs, rhs).apart)
    for lhs, rhs in [(o1h, o2a), (o1a, o2h)]:
        if lhs.num_terms * rhs.num_terms != 0:
            comms.append(2. * spv_prod(lhs, rhs).hpart)
    return spv_sum(*comms)


def spvlist_to_csc(ops: Sequence[SparsePauliVector]) -> csc_array:
    if len(ops) == 0:
        raise ValueError('Cannot convert zero-length list to CSC')

    shape = (ops[0].vlen, len(ops))
    data = np.concatenate([op.coeffs for op in ops])
    indices = np.concatenate([op.indices for op in ops])
    indptr = np.cumsum([0] + [op.num_terms for op in ops])
    return csc_array((data, indices, indptr), shape=shape)


def csc_to_spvlist(matrix: csc_array) -> list[SparsePauliVector]:
    num_qubits = int(np.round(np.emath.logn(4, matrix.shape[0])))
    ops = []
    for icol in range(matrix.shape[1]):
        idx_begin, idx_end = matrix.indptr[icol:icol+2]
        ops.append(SparsePauliVector(matrix.indices[idx_begin:idx_end],
                                     matrix.data[idx_begin:idx_end],
                                     num_qubits=num_qubits))

    return ops
