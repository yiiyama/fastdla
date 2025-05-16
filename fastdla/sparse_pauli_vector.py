"""Purpose-specific implementation of a sparse Pauli operator."""
from collections.abc import Sequence
from numbers import Number
from typing import Optional
import numpy as np
from scipy.sparse import csr_array

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
PAULIS = np.array([
    [[1., 0.], [0., 1.]],
    [[0., 1.], [1., 0.]],
    [[0., -1.j], [1.j, 0.]],
    [[1., 0.], [0., -1.]]
])


class SparsePauliVector:
    """Purpose-specific implementation of a sparse Pauli operator."""
    @staticmethod
    def str_to_idx(pstr: str) -> int:
        indices = np.array([PAULI_INDICES[p] for p in pstr], dtype=np.uint64)
        return np.sum(indices * (4 ** np.arange(indices.shape[0])[::-1]))

    @staticmethod
    def idx_to_str(idx: int, num_qubits: int) -> str:
        indices = ((idx // (4 ** np.arange(num_qubits)[::-1])) % 4).astype(int)
        return ''.join(PAULI_NAMES[i] for i in indices)

    @classmethod
    def switch_impl(cls, to: str):
        if to == 'ref':
            cls.__add__ = spv_sum
            cls.__sub__ = lambda self, other: spv_sum(self, -other)
            cls.__matmul__ = spv_prod
            cls.commutator = spv_commutator
            cls.dot = spv_innerprod
        elif to == 'fast':
            # pylint: disable-next=import-outside-toplevel
            from fastdla.spv_fast import (spv_sum_fast, spv_prod_fast, spv_commutator_fast,
                                          spv_innerprod_fast)
            cls.__add__ = spv_sum_fast
            cls.__sub__ = lambda self, other: spv_sum_fast(self, -other)
            cls.__matmul__ = spv_prod_fast
            cls.commutator = spv_commutator_fast
            cls.dot = spv_innerprod_fast

    def __init__(
        self,
        indices: str | int | Sequence[str] | Sequence[int],
        coeffs: Number | Sequence[Number],
        num_qubits: Optional[int] = None,
        no_check: bool = False
    ):
        if no_check:
            self.indices = np.asarray(indices, dtype=np.uint64)
            self.coeffs = np.asarray(coeffs, dtype=np.complex128)
            self.num_qubits = num_qubits
            return

        if isinstance(indices, (str, int)):
            indices = [indices]
        if isinstance(coeffs, Number):
            coeffs = [coeffs]

        if len(indices) == 0:
            self.num_qubits = num_qubits
            self.indices = np.array([], dtype=np.uint64)
            self.coeffs = np.array([], dtype=np.complex128)
            return

        if isinstance(indices[0], str):
            num_qubits = len(indices[0])
            indices = np.array([self.str_to_idx(pstr) for pstr in indices], dtype=np.uint64)
        elif num_qubits is None:
            raise ValueError('Need num_qubits')
        else:
            indices = np.asarray(indices, dtype=np.uint64)

        self.num_qubits = num_qubits
        sort_idx = np.argsort(indices)
        self.indices = indices[sort_idx]
        self.coeffs = np.asarray(coeffs, dtype=np.complex128)[sort_idx]

        if self.indices.shape != self.coeffs.shape:
            raise ValueError('Indices and coeffs shape mismatch')

    def __str__(self) -> str:
        opstr = ', '.join(f'{coeff} {self.idx_to_str(idx, self.num_qubits)}'
                          for coeff, idx in zip(self.coeffs, self.indices))
        return f'[{opstr}]'

    def __repr__(self) -> str:
        return f'SparsePauliVector({repr(self.paulis)}, {repr(self.coeffs)})'

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
    def paulis(self) -> list[str]:
        return [self.idx_to_str(idx, self.num_qubits) for idx in self.indices]

    @property
    def hpart(self) -> 'SparsePauliVector':
        """Return a new SPO with Hermitian terms only."""
        indices = np.nonzero(np.logical_not(np.isclose(self.coeffs.real, 0.)))[0]
        return SparsePauliVector(self.indices[indices], self.coeffs[indices].real, self.num_qubits,
                                 no_check=True)

    @property
    def apart(self) -> 'SparsePauliVector':
        """Return a new SPO with anti-Hermitian terms only."""
        indices = np.nonzero(np.logical_not(np.isclose(self.coeffs.imag, 0.)))[0]
        return SparsePauliVector(self.indices[indices], 1.j * self.coeffs[indices].imag,
                                 self.num_qubits, no_check=True)

    def normalize(self) -> 'SparsePauliVector':
        norm = np.sqrt(np.sum(np.square(np.abs(self.coeffs))))
        return SparsePauliVector(self.indices.copy(), self.coeffs / norm, self.num_qubits,
                                 no_check=True)

    def __eq__(self, other: 'SparsePauliVector') -> bool:
        if self.num_qubits != other.num_qubits or self.num_terms != other.num_terms:
            return False
        if np.any(self.indices != other.indices):
            return False
        return np.allclose(self.coeffs, other.coeffs)

    def __neg__(self) -> 'SparsePauliVector':
        return SparsePauliVector(self.indices, -self.coeffs, self.num_qubits, no_check=True)

    def __add__(self, other: 'SparsePauliVector') -> 'SparsePauliVector':
        return spv_sum(self, other)

    def __sub__(self, other: 'SparsePauliVector') -> 'SparsePauliVector':
        return spv_sum(self, -other)

    def __mul__(self, scalar: Number) -> 'SparsePauliVector':
        try:
            coeffs = scalar * self.coeffs
        except Exception:  # pylint: disable=broad-exception-caught
            return NotImplemented
        return SparsePauliVector(self.indices, coeffs, self.num_qubits, no_check=True)

    def __rmul__(self, scalar: Number) -> 'SparsePauliVector':
        return self.__mul__(scalar)

    def __matmul__(self, other: 'SparsePauliVector') -> 'SparsePauliVector':
        return spv_prod(self, other)

    def commutator(self, other: 'SparsePauliVector') -> 'SparsePauliVector':
        return spv_commutator(self, other)

    def dot(self, other: 'SparsePauliVector') -> complex:
        return spv_innerprod(self, other)

    def to_csr(self) -> csr_array:
        shape = (1, self.vlen)
        return csr_array((self.coeffs, self.indices, [0, self.num_terms]), shape=shape)

    @staticmethod
    def from_csr(array: csr_array, num_qubits: int) -> 'SparsePauliVector':
        if array.shape[0] != 1:
            raise ValueError('Only a single-row array can be converted to SPV')
        return SparsePauliVector(array.indices, array.data, num_qubits, no_check=True)

    def to_dense(self) -> np.ndarray:
        dense = np.zeros(self.vlen, dtype=self.coeffs.dtype)
        dense[self.indices] = self.coeffs
        return dense

    def to_matrix(self, *, npmod=np) -> np.ndarray:
        """Convert the Pauli sum to a dense (2**num_qubits, 2**num_qubits) matrix.

        SparsePauliOp.to_matrix() seems to have a much more efficient implementation.
        """
        pauli_indices = ((self.indices[:, None] // (4 ** np.arange(self.num_qubits)[None, ::-1]))
                         % 4).astype(int)
        args = ()
        for iq in range(self.num_qubits):
            args += (PAULIS[pauli_indices[:, iq]], [0, iq + 1, self.num_qubits + iq + 1])
        args += (list(range(2 * self.num_qubits + 1)),)
        matrices = npmod.einsum(*args)
        matrices = matrices.reshape((self.num_terms, 2 ** self.num_qubits, 2 ** self.num_qubits))
        return npmod.sum(self.coeffs[:, None, None] * matrices, axis=0)


class SparsePauliVectorArray:
    """A container of SparsePauliVectors with concatenated data."""
    MEM_ALLOC_UNIT = 1024

    def __init__(
        self,
        vectors: Optional[list[SparsePauliVector]] = None,
        initial_capacity: int = MEM_ALLOC_UNIT,
    ):
        initial_capacity = (((initial_capacity - 1) // self.MEM_ALLOC_UNIT + 1)
                            * self.MEM_ALLOC_UNIT)
        self._indices = np.empty(initial_capacity, dtype=np.uint64)
        self._coeffs = np.empty(initial_capacity, dtype=np.complex128)
        self._ptrs = [0]

        self.num_qubits = None

        for vector in (vectors or []):
            self.append(vector)

    @property
    def shape(self) -> tuple[int]:
        return (len(self),) + (4,) * self.num_qubits

    @property
    def vlen(self) -> int:
        return 4**self.num_qubits

    def _expand(self, new_capacity: int):
        if (addition := new_capacity - self._indices.shape[0]) <= 0:
            return
        self._indices = np.concatenate([self._indices, np.empty(addition, dtype=np.uint64)])
        self._coeffs = np.concatenate([self._coeffs, np.empty(addition, dtype=np.complex128)])

    def append(self, vector: SparsePauliVector):
        if self.num_qubits is None:
            self.num_qubits = vector.num_qubits
        elif vector.num_qubits != self.num_qubits:
            raise ValueError('Inconsistent num_qubits')

        if (new_end := self._ptrs[-1] + vector.num_terms) > self._indices.shape[0]:
            self._expand((new_end // self.MEM_ALLOC_UNIT + 1) * self.MEM_ALLOC_UNIT)

        self._ptrs.append(self._ptrs[-1] + vector.num_terms)
        self._indices[self._ptrs[-2]:self._ptrs[-1]] = vector.indices
        self._coeffs[self._ptrs[-2]:self._ptrs[-1]] = vector.coeffs

    def __len__(self) -> int:
        return len(self._ptrs) - 1

    def __getitem__(self, idx: int) -> SparsePauliVector:
        if idx < 0 or idx >= len(self._ptrs) - 1:
            raise IndexError(f'Invalid vector index {idx}')

        return SparsePauliVector(
            self._indices[self._ptrs[idx]:self._ptrs[idx + 1]],
            self._coeffs[self._ptrs[idx]:self._ptrs[idx + 1]],
            self.num_qubits,
            no_check=True
        )

    def to_csr(self) -> csr_array:
        return csr_array((self._coeffs, self._indices, self._ptrs),
                         shape=(len(self), self.vlen))

    @staticmethod
    def from_csr(matrix: csr_array) -> 'SparsePauliVectorArray':
        array = SparsePauliVectorArray(initial_capacity=matrix.data.shape[0])
        array.num_qubits = int(np.round(np.emath.logn(4, matrix.shape[1])))
        array._indices = matrix.indices
        array._coeffs = matrix.data
        array._ptrs = list(matrix.indptr)
        return array


def _uniquify(indices: np.ndarray, coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Uniquify a pauli list by summing up the cofficients of overlapping operators."""
    indices_uniq, unique_indices = np.unique(indices, return_inverse=True)
    if indices_uniq.shape[0] == indices.shape[0]:
        # indices is already unique
        return indices_uniq, coeffs[np.argsort(unique_indices)]

    # indices is not unique -> merge duplicates
    coeffs_matrix = np.zeros((coeffs.shape[0], indices_uniq.shape[0]), dtype=coeffs.dtype)
    coeffs_matrix[np.arange(coeffs.shape[0]), unique_indices] = coeffs
    coeffs_uniq = np.sum(coeffs_matrix, axis=0)
    # remove terms with null coefficients
    nonzero = np.nonzero(np.logical_not(np.isclose(coeffs_uniq, 0.)))[0]
    return indices_uniq[nonzero], coeffs_uniq[nonzero]


def spv_sum(*ops) -> SparsePauliVector:
    """Sum of sparse Pauli ops."""
    indices, coeffs = _uniquify(
        np.concatenate([op.indices for op in ops]),
        np.concatenate([op.coeffs for op in ops])
    )
    return SparsePauliVector(indices, coeffs, ops[0].num_qubits, no_check=True)


def spv_prod(o1: SparsePauliVector, o2: SparsePauliVector) -> SparsePauliVector:
    """Product of two sparse Pauli ops."""
    if o1.num_qubits != o2.num_qubits:
        raise ValueError('Matmul between incompatible SparsePauliOps')

    if (num_terms := o1.num_terms * o2.num_terms) == 0:
        return SparsePauliVector([], [], o1.num_qubits, no_check=True)

    indices = np.unravel_index(np.arange(num_terms), (o1.num_terms, o2.num_terms))
    coeffs = np.outer(o1.coeffs, o2.coeffs).reshape(-1)
    s1s = o1.indices[indices[0]]
    s2s = o2.indices[indices[1]]

    p1s = ((s1s[:, None] // (4 ** np.arange(o1.num_qubits)[None, ::-1])) % 4).astype(int)
    p2s = ((s2s[:, None] // (4 ** np.arange(o2.num_qubits)[None, ::-1])) % 4).astype(int)
    pout = PAULI_PROD_INDEX[p1s, p2s]  # shape [num_terms, num_qubits]
    sout = np.sum(pout * (4 ** np.arange(o1.num_qubits)[None, ::-1]), axis=1)
    coeffs *= np.prod(PAULI_PROD_COEFF[p1s, p2s], axis=1)

    indices, coeffs = _uniquify(sout, coeffs)
    return SparsePauliVector(indices, coeffs, o1.num_qubits, no_check=True)


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


def spv_innerprod(o1: SparsePauliVector, o2: SparsePauliVector) -> complex:
    common_entries = np.nonzero(o1.indices[:, None] - o2.indices[None, :] == 0)
    return np.sum(np.conjugate(o1.data[common_entries[0]]) * o2.data[common_entries[1]])
