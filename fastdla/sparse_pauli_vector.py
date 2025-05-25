"""Purpose-specific implementation of a sparse Pauli operator."""
from collections.abc import Sequence
from numbers import Number
from typing import Optional
import numpy as np
from scipy.sparse import csr_array
from .pauli import PauliProduct


class SparsePauliVector:
    """Purpose-specific implementation of a sparse Pauli operator."""
    @classmethod
    def switch_impl(cls, to: str):
        if to == 'ref':
            cls.__add__ = spv_add
            cls.__sub__ = lambda self, other: spv_add(self, -other)
            cls.__matmul__ = spv_matmul
            cls.commutator = spv_commutator
            cls.dot = spv_dot
        elif to == 'fast':
            # pylint: disable-next=import-outside-toplevel
            from fastdla.spv_fast import (spv_add_fast, spv_matmul_fast, spv_commutator_fast,
                                          spv_dot_fast)
            cls.__add__ = spv_add_fast
            cls.__sub__ = lambda self, other: spv_add_fast(self, -other)
            cls.__matmul__ = spv_matmul_fast
            cls.commutator = spv_commutator_fast
            cls.dot = spv_dot_fast

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
            indices = np.array([PauliProduct.str_to_idx(pstr) for pstr in indices], dtype=np.uint64)
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
        opstr = ', '.join(f'{coeff} {str(PauliProduct(idx, self.num_qubits))}'
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
        return [str(PauliProduct(idx, self.num_qubits)) for idx in self.indices]

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
        return spv_add(self, other)

    def __sub__(self, other: 'SparsePauliVector') -> 'SparsePauliVector':
        return spv_add(self, -other)

    def __mul__(self, scalar: Number) -> 'SparsePauliVector':
        try:
            coeffs = scalar * self.coeffs
        except Exception:  # pylint: disable=broad-exception-caught
            return NotImplemented
        return SparsePauliVector(self.indices, coeffs, self.num_qubits, no_check=True)

    def __rmul__(self, scalar: Number) -> 'SparsePauliVector':
        return self.__mul__(scalar)

    def __matmul__(self, other: 'SparsePauliVector') -> 'SparsePauliVector':
        return spv_matmul(self, other)

    def commutator(
        self,
        other: 'SparsePauliVector',
        normalize: bool = False
    ) -> 'SparsePauliVector':
        return spv_commutator(self, other, normalize)

    def dot(self, other: 'SparsePauliVector') -> complex:
        return spv_dot(self, other)

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

    def to_matrix(self, *, sparse: bool = False, npmod=np) -> np.ndarray:
        """Convert the Pauli sum to a dense (2**num_qubits, 2**num_qubits) matrix.

        SparsePauliOp.to_matrix() seems to have a much more efficient implementation.
        """
        matrix = (
            PauliProduct(self.indices[0], self.num_qubits).to_matrix(sparse=sparse, npmod=npmod)
            * self.coeffs[0]
        )
        for index, coeff in zip(self.indices[1:], self.coeffs[1:]):
            matrix += (PauliProduct(index, self.num_qubits).to_matrix(sparse=sparse, npmod=npmod)
                       * coeff)
        return matrix


class SparsePauliVectorArray:
    """A container of SparsePauliVectors with concatenated data."""
    MEM_ALLOC_UNIT = 1024

    def __init__(
        self,
        vectors: Optional[list[SparsePauliVector]] = None,
        num_qubits: Optional[int] = None,
        initial_capacity: int = MEM_ALLOC_UNIT,
    ):
        initial_capacity = (((initial_capacity - 1) // self.MEM_ALLOC_UNIT + 1)
                            * self.MEM_ALLOC_UNIT)
        self.indices = np.empty(initial_capacity, dtype=np.uint64)
        self.coeffs = np.empty(initial_capacity, dtype=np.complex128)
        self.ptrs = [0]
        self.num_qubits = num_qubits

        for vector in (vectors or []):
            self.append(vector)

    def __str__(self) -> str:
        opstr = '[\n'
        opstr += ',\n'.join(f'  {str(self[i])}' for i in range(min(4, len(self))))
        if len(self) > 4:
            opstr += ',\n...,'
        opstr += '\n]'
        return opstr

    def __repr__(self) -> str:
        return f'SparsePauliVectorArray({len(self)} vectors, num_qubits={self.num_qubits})'

    @property
    def shape(self) -> tuple[int]:
        return (len(self),) + (4,) * self.num_qubits

    @property
    def vlen(self) -> int:
        return 4**self.num_qubits

    @property
    def capacity(self) -> int:
        return self.indices.shape[0]

    def _expand(self, new_capacity: int):
        if (addition := new_capacity - self.indices.shape[0]) <= 0:
            return
        self.indices = np.concatenate([self.indices, np.empty(addition, dtype=np.uint64)])
        self.coeffs = np.concatenate([self.coeffs, np.empty(addition, dtype=np.complex128)])

    def append(self, vector: SparsePauliVector):
        if self.num_qubits is None:
            self.num_qubits = vector.num_qubits
        elif vector.num_qubits != self.num_qubits:
            raise ValueError('Inconsistent num_qubits')

        if (new_end := self.ptrs[-1] + vector.num_terms) > self.indices.shape[0]:
            self._expand((new_end // self.MEM_ALLOC_UNIT + 1) * self.MEM_ALLOC_UNIT)

        self.ptrs.append(self.ptrs[-1] + vector.num_terms)
        self.indices[self.ptrs[-2]:self.ptrs[-1]] = vector.indices
        self.coeffs[self.ptrs[-2]:self.ptrs[-1]] = vector.coeffs

    def normalize(self) -> 'SparsePauliVectorArray':
        normalized = SparsePauliVectorArray(initial_capacity=self.capacity)
        normalized.num_qubits = self.num_qubits
        normalized.indices = self.indices.copy()
        normalized.ptrs = self.ptrs.copy()
        for start, end in zip(self.ptrs[:-1], self.ptrs[1:]):
            norm = np.sqrt(np.sum(np.square(np.abs(self.coeffs[start:end]))))
            normalized.coeffs[start:end] = self.coeffs[start:end] / norm
        return normalized

    def __len__(self) -> int:
        return len(self.ptrs) - 1

    def __getitem__(self, idx: int | slice) -> SparsePauliVector:
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f'Invalid vector index {idx}')

        return SparsePauliVector(
            self.indices[self.ptrs[idx]:self.ptrs[idx + 1]],
            self.coeffs[self.ptrs[idx]:self.ptrs[idx + 1]],
            self.num_qubits,
            no_check=True
        )

    def to_csr(self) -> csr_array:
        return csr_array((self.coeffs, self.indices, self.ptrs),
                         shape=(len(self), self.vlen))

    @staticmethod
    def from_csr(matrix: csr_array) -> 'SparsePauliVectorArray':
        array = SparsePauliVectorArray(initial_capacity=matrix.data.shape[0])
        array.num_qubits = int(np.round(np.emath.logn(4, matrix.shape[1])))
        array.indices = matrix.indices
        array.coeffs = matrix.data
        array.ptrs = list(matrix.indptr)
        return array

    def to_matrices(self, *, sparse: bool = False, npmod=np) -> np.ndarray:
        matrices = npmod.empty((len(self), 2 ** self.num_qubits, 2 ** self.num_qubits),
                               dtype=np.complex128)
        for ivec, vec in enumerate(self):
            matrices[ivec] = vec.to_matrix(sparse=sparse, npmod=npmod)
        return matrices


def _uniquify(
    indices: np.ndarray,
    coeffs: np.ndarray,
    normalize: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Uniquify a pauli list by summing up the cofficients of overlapping operators."""
    indices_uniq, unique_indices = np.unique(indices, return_inverse=True)
    if indices_uniq.shape[0] == indices.shape[0]:
        # indices is already unique
        coeffs_uniq = coeffs[np.argsort(unique_indices)]
    else:
        # indices is not unique -> merge duplicates
        coeffs_matrix = np.zeros((coeffs.shape[0], indices_uniq.shape[0]), dtype=coeffs.dtype)
        coeffs_matrix[np.arange(coeffs.shape[0]), unique_indices] = coeffs
        coeffs_uniq = np.sum(coeffs_matrix, axis=0)
        # remove terms with null coefficients
        nonzero = np.nonzero(np.logical_not(np.isclose(coeffs_uniq, 0.)))[0]
        indices_uniq = indices_uniq[nonzero]
        coeffs_uniq = coeffs_uniq[nonzero]

    if normalize:
        coeffs_uniq /= np.sqrt(np.sum(np.square(np.abs(coeffs_uniq))))

    return indices_uniq, coeffs_uniq


def spv_add(*ops, normalize: bool = False) -> SparsePauliVector:
    """Sum of sparse Pauli ops."""
    indices, coeffs = _uniquify(
        np.concatenate([op.indices for op in ops]),
        np.concatenate([op.coeffs for op in ops]),
        normalize
    )
    return SparsePauliVector(indices, coeffs, ops[0].num_qubits, no_check=True)


def spv_matmul(
    o1: SparsePauliVector,
    o2: SparsePauliVector,
    normalize: bool = False
) -> SparsePauliVector:
    """Product of two sparse Pauli ops."""
    if (num_qubits := o1.num_qubits) != o2.num_qubits:
        raise ValueError('Matmul between incompatible SparsePauliVectors')

    if (num_terms := o1.num_terms * o2.num_terms) == 0:
        return SparsePauliVector([], [], num_qubits, no_check=True)

    indices = np.unravel_index(np.arange(num_terms), (o1.num_terms, o2.num_terms))
    coeffs = np.outer(o1.coeffs, o2.coeffs).reshape(-1)
    i1 = o1.indices[indices[0]]
    c1 = o1.coeffs[indices[0]]
    i2 = o2.indices[indices[1]]
    c2 = o2.coeffs[indices[1]]
    indices, coeffs = PauliProduct.matmul(i1, c1, i2, c2, num_qubits)
    indices, coeffs = _uniquify(indices, coeffs, normalize)
    return SparsePauliVector(indices, coeffs, num_qubits, no_check=True)


def spv_commutator(
    o1: SparsePauliVector,
    o2: SparsePauliVector,
    normalize: bool = False
) -> SparsePauliVector:
    o1h = o1.hpart
    o1a = o1.apart
    o2h = o2.hpart
    o2a = o2.apart
    comms = []
    for lhs, rhs in [(o1h, o2h), (o1a, o2a)]:
        if lhs.num_terms * rhs.num_terms != 0:
            comms.append(2. * spv_matmul(lhs, rhs).apart)
    for lhs, rhs in [(o1h, o2a), (o1a, o2h)]:
        if lhs.num_terms * rhs.num_terms != 0:
            comms.append(2. * spv_matmul(lhs, rhs).hpart)
    return spv_add(*comms, normalize=normalize)


def spv_dot(o1: SparsePauliVector, o2: SparsePauliVector) -> complex:
    common_entries = np.nonzero(o1.indices[:, None] - o2.indices[None, :] == 0)
    return np.sum(o1.coeffs[common_entries[0]].conjugate() * o2.coeffs[common_entries[1]])
