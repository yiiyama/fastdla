"""Purpose-specific implementation of a sparse Pauli operator."""
from collections.abc import Sequence
from numbers import Number
from typing import Optional
import numpy as np
from scipy.sparse import csr_array
from .pauli import PauliProduct


class SparsePauliSum:
    """Purpose-specific implementation of a sparse Pauli operator.

    ``SparsePauliSum`` is conceptually an array of ``PauliProduct``s with complex coefficients.
    Internally, the Pauli product indices and coefficients are stored separately as numpy arrays.

    Args:
        indices: A list of parameters, each of which can initialize a PauliProduct. A single integer
            or str is allowed, in which case a length-1 Pauli sum is initialized.
        coeffs: Array of coefficients corresponding to the Pauli products specified by ``indices``.
        num_qubits: Required if Pauli products are specified using the compact index.
        no_check: If True, ``indices`` is assumed to be an array of compact indices, and no
            formatting will be performed.
    """
    @classmethod
    def switch_impl(cls, to: str):
        """Switch the implementation of operators.

        The options are

        * 'ref': Reference implementation using numpy functions.
        * 'fast': Numba-based implementation.

        Args:
            to: Name of the implementation to switch to.
        """
        if to == 'ref':
            cls.__add__ = sps_add
            cls.__sub__ = lambda self, other: sps_add(self, -other)
            cls.__matmul__ = sps_matmul
            cls.commutator = sps_commutator
            cls.dot = sps_dot
        elif to == 'fast':
            # pylint: disable-next=import-outside-toplevel
            from fastdla.sps_fast import (sps_add_fast, sps_matmul_fast, sps_commutator_fast,
                                          sps_dot_fast)
            cls.__add__ = sps_add_fast
            cls.__sub__ = lambda self, other: sps_add_fast(self, -other)
            cls.__matmul__ = sps_matmul_fast
            cls.commutator = sps_commutator_fast
            cls.dot = sps_dot_fast

    def __init__(
        self,
        indices: str | int | Sequence[str] | Sequence[Sequence[int]] | Sequence[int],
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
        elif isinstance(indices[0], Sequence):
            indices = np.array([PauliProduct.seq_to_idx(plst) for plst in indices], dtype=np.uint64)
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
        return f'SparsePauliSum({repr(self.paulis)}, {repr(self.coeffs)})'

    @property
    def shape(self) -> tuple[int]:
        """Shape of the Pauli tensor."""
        return (4,) * self.num_qubits

    @property
    def vlen(self) -> int:
        """Length of the Pauli vector."""
        return 4**self.num_qubits

    @property
    def num_terms(self) -> int:
        """Number of terms in the sum."""
        return self.indices.shape[0]

    @property
    def paulis(self) -> list[str]:
        """Return the list of PauliProducts in the string representation."""
        return [str(PauliProduct(idx, self.num_qubits)) for idx in self.indices]

    @property
    def hpart(self) -> 'SparsePauliSum':
        """Return a new SPS with Hermitian terms only."""
        indices = np.nonzero(np.logical_not(np.isclose(self.coeffs.real, 0.)))[0]
        return SparsePauliSum(self.indices[indices], self.coeffs[indices].real, self.num_qubits,
                              no_check=True)

    @property
    def apart(self) -> 'SparsePauliSum':
        """Return a new SPS with anti-Hermitian terms only."""
        indices = np.nonzero(np.logical_not(np.isclose(self.coeffs.imag, 0.)))[0]
        return SparsePauliSum(self.indices[indices], 1.j * self.coeffs[indices].imag,
                              self.num_qubits, no_check=True)

    def normalize(self) -> 'SparsePauliSum':
        """Return a normalized copy of this Pauli sum."""
        norm = np.sqrt(np.sum(np.square(np.abs(self.coeffs))))
        return SparsePauliSum(self.indices.copy(), self.coeffs / norm, self.num_qubits,
                              no_check=True)

    def __eq__(self, other: 'SparsePauliSum') -> bool:
        if self.num_qubits != other.num_qubits or self.num_terms != other.num_terms:
            return False
        if np.any(self.indices != other.indices):
            return False
        return np.allclose(self.coeffs, other.coeffs)

    def __neg__(self) -> 'SparsePauliSum':
        return SparsePauliSum(self.indices, -self.coeffs, self.num_qubits, no_check=True)

    def __add__(self, other: 'SparsePauliSum') -> 'SparsePauliSum':
        return sps_add(self, other)

    def __sub__(self, other: 'SparsePauliSum') -> 'SparsePauliSum':
        return sps_add(self, -other)

    def __mul__(self, scalar: Number) -> 'SparsePauliSum':
        try:
            coeffs = scalar * self.coeffs
        except Exception:  # pylint: disable=broad-exception-caught
            return NotImplemented
        return SparsePauliSum(self.indices, coeffs, self.num_qubits, no_check=True)

    def __rmul__(self, scalar: Number) -> 'SparsePauliSum':
        return self.__mul__(scalar)

    def __matmul__(self, other: 'SparsePauliSum') -> 'SparsePauliSum':
        return sps_matmul(self, other)

    def commutator(
        self,
        other: 'SparsePauliSum',
        normalize: bool = False
    ) -> 'SparsePauliSum':
        """Return the commutator [self, other]."""
        return sps_commutator(self, other, normalize)

    def dot(self, other: 'SparsePauliSum') -> complex:
        """Return the inner product <self, other>."""
        return sps_dot(self, other)

    def to_csr(self) -> csr_array:
        """Convert to a sparse Pauli vector in the CSR format."""
        shape = (1, self.vlen)
        return csr_array((self.coeffs, self.indices, [0, self.num_terms]), shape=shape)

    @staticmethod
    def from_csr(array: csr_array, num_qubits: int) -> 'SparsePauliSum':
        """Convert from a sparse Pauli vector in the CSR format."""
        if array.shape[0] != 1:
            raise ValueError('Only a single-row array can be converted to a SparsePauliSum')
        return SparsePauliSum(array.indices, array.data, num_qubits, no_check=True)

    def to_dense(self) -> np.ndarray:
        """Convert to a dense Pauli vector."""
        dense = np.zeros(self.vlen, dtype=self.coeffs.dtype)
        dense[self.indices] = self.coeffs
        return dense

    def to_matrix(self, *, sparse: bool = False, npmod=np) -> np.ndarray | csr_array:
        # pylint: disable=import-outside-toplevel
        """Convert the Pauli sum to a dense (2**num_qubits, 2**num_qubits) matrix."""
        matrix = sum(
            PauliProduct(index, self.num_qubits).to_matrix(sparse=sparse) * coeff
            for index, coeff in zip(self.indices, self.coeffs)
        )
        if not sparse or npmod is np:
            return matrix

        import jax.numpy as jnp
        from jax.experimental.sparse import BCSR
        if npmod is jnp:
            return BCSR.from_scipy_sparse(matrix)


class SparsePauliSumArray:
    """A container of SparsePauliSums with concatenated data.

    Conceptually, ``SparsePauliSumArray`` is simply a list of ``SparsePauliSum``s, and have
    interfaces that resemble a list (such as ``append``, ``__getitem__``, and ``__len__``).
    Internally, the indices and coeffs are directly concatenated into single arrays.

    Args:
        pauli_sums: List of SparsePauliSums.
        num_qubits: Used when initializing a length-0 array. If not given, num_qubits of the first
            SparsePauliSum to be appended will be used.
        initial_capacity: Initial maximum size of the concatenated arrays.
    """
    MEM_ALLOC_UNIT = 1024

    def __init__(
        self,
        pauli_sums: Optional[list[SparsePauliSum]] = None,
        num_qubits: Optional[int] = None,
        initial_capacity: int = MEM_ALLOC_UNIT,
    ):
        initial_capacity = (((initial_capacity - 1) // self.MEM_ALLOC_UNIT + 1)
                            * self.MEM_ALLOC_UNIT)
        self.indices = np.empty(initial_capacity, dtype=np.uint64)
        self.coeffs = np.empty(initial_capacity, dtype=np.complex128)
        self.ptrs = [0]
        self.num_qubits = num_qubits

        for pauli_sum in (pauli_sums or []):
            self.append(pauli_sum)

    def __str__(self) -> str:
        opstr = '[\n'
        opstr += ',\n'.join(f'  {str(self[i])}' for i in range(min(4, len(self))))
        if len(self) > 4:
            opstr += ',\n...,'
        opstr += '\n]'
        return opstr

    def __repr__(self) -> str:
        return f'SparsePauliSumArray({len(self)} elements, num_qubits={self.num_qubits})'

    @property
    def shape(self) -> tuple[int]:
        """Shape of the array as a list of Pauli tensors."""
        return (len(self),) + (4,) * self.num_qubits

    @property
    def vlen(self) -> int:
        """Length of the component Pauli vectors."""
        return 4**self.num_qubits

    @property
    def capacity(self) -> int:
        """Current maximum size of the concatenated arrays."""
        return self.indices.shape[0]

    def _expand(self, new_capacity: int):
        if (addition := new_capacity - self.indices.shape[0]) <= 0:
            return
        self.indices = np.concatenate([self.indices, np.empty(addition, dtype=np.uint64)])
        self.coeffs = np.concatenate([self.coeffs, np.empty(addition, dtype=np.complex128)])

    def append(self, pauli_sum: SparsePauliSum):
        """Append a new SparsePauliSum."""
        if self.num_qubits is None:
            self.num_qubits = pauli_sum.num_qubits
        elif pauli_sum.num_qubits != self.num_qubits:
            raise ValueError('Inconsistent num_qubits')

        if (new_end := self.ptrs[-1] + pauli_sum.num_terms) > self.indices.shape[0]:
            self._expand((new_end // self.MEM_ALLOC_UNIT + 1) * self.MEM_ALLOC_UNIT)

        self.ptrs.append(self.ptrs[-1] + pauli_sum.num_terms)
        self.indices[self.ptrs[-2]:self.ptrs[-1]] = pauli_sum.indices
        self.coeffs[self.ptrs[-2]:self.ptrs[-1]] = pauli_sum.coeffs

    def normalize(self) -> 'SparsePauliSumArray':
        """Return a new array with normalized elements."""
        normalized = SparsePauliSumArray(initial_capacity=self.capacity)
        normalized.num_qubits = self.num_qubits
        normalized.indices = self.indices.copy()
        normalized.ptrs = self.ptrs.copy()
        for start, end in zip(self.ptrs[:-1], self.ptrs[1:]):
            norm = np.sqrt(np.sum(np.square(np.abs(self.coeffs[start:end]))))
            normalized.coeffs[start:end] = self.coeffs[start:end] / norm
        return normalized

    def __len__(self) -> int:
        return len(self.ptrs) - 1

    def __getitem__(self, idx: int | slice) -> SparsePauliSum:
        if isinstance(idx, slice):
            return SparsePauliSumArray([self[i] for i in range(*idx.indices(len(self)))])
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f'Invalid index {idx}')

        return SparsePauliSum(
            self.indices[self.ptrs[idx]:self.ptrs[idx + 1]],
            self.coeffs[self.ptrs[idx]:self.ptrs[idx + 1]],
            self.num_qubits,
            no_check=True
        )

    def to_csr(self) -> csr_array:
        """Convert to an array of Pauli vectors in the CSR format."""
        return csr_array((self.coeffs, self.indices, self.ptrs),
                         shape=(len(self), self.vlen))

    @staticmethod
    def from_csr(matrix: csr_array) -> 'SparsePauliSumArray':
        """Convert from an array of Pauli vectors in the CSR format."""
        array = SparsePauliSumArray(initial_capacity=matrix.data.shape[0])
        array.num_qubits = int(np.round(np.emath.logn(4, matrix.shape[1])))
        array.indices = matrix.indices
        array.coeffs = matrix.data
        array.ptrs = list(matrix.indptr)
        return array

    def to_matrices(self, *, sparse: bool = False, npmod=np) -> np.ndarray | list[csr_array]:
        """Convert to an array of dense matrices."""
        if sparse:
            matrices = [elem.to_matrix(sparse=True, npmod=npmod) for elem in self]
        else:
            matrices = npmod.empty((len(self), 2 ** self.num_qubits, 2 ** self.num_qubits),
                                   dtype=np.complex128)
            if npmod is np:
                for ielem, elem in enumerate(self):
                    matrices[ielem] = elem.to_matrix(sparse=False)
            else:
                for ielem, elem in enumerate(self):
                    matrices = matrices.at[ielem].set(elem.to_matrix(sparse=False, npmod=npmod))

        return matrices


def _uniquify(
    indices: np.ndarray,
    coeffs: np.ndarray,
    normalize: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Uniquify a Pauli list by summing up the cofficients of overlapping operators."""
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


def sps_add(*ops, normalize: bool = False) -> SparsePauliSum:
    """Sum of SparsePauliSums."""
    indices, coeffs = _uniquify(
        np.concatenate([op.indices for op in ops]),
        np.concatenate([op.coeffs for op in ops]),
        normalize
    )
    return SparsePauliSum(indices, coeffs, ops[0].num_qubits, no_check=True)


def sps_matmul(
    o1: SparsePauliSum,
    o2: SparsePauliSum,
    normalize: bool = False
) -> SparsePauliSum:
    """Product of two SparsePauliSums."""
    if (num_qubits := o1.num_qubits) != o2.num_qubits:
        raise ValueError('Matmul between incompatible SparsePauliSums')

    if (num_terms := o1.num_terms * o2.num_terms) == 0:
        return SparsePauliSum([], [], num_qubits, no_check=True)

    indices = np.unravel_index(np.arange(num_terms), (o1.num_terms, o2.num_terms))
    coeffs = np.outer(o1.coeffs, o2.coeffs).reshape(-1)
    i1 = o1.indices[indices[0]]
    c1 = o1.coeffs[indices[0]]
    i2 = o2.indices[indices[1]]
    c2 = o2.coeffs[indices[1]]
    indices, coeffs = PauliProduct.matmul(i1, c1, i2, c2, num_qubits)
    indices, coeffs = _uniquify(indices, coeffs, normalize)
    return SparsePauliSum(indices, coeffs, num_qubits, no_check=True)


def sps_commutator(
    o1: SparsePauliSum,
    o2: SparsePauliSum,
    normalize: bool = False
) -> SparsePauliSum:
    """Commutator between two SparsePauliSums."""
    o1h = o1.hpart
    o1a = o1.apart
    o2h = o2.hpart
    o2a = o2.apart
    comms = []
    for lhs, rhs in [(o1h, o2h), (o1a, o2a)]:
        if lhs.num_terms * rhs.num_terms != 0:
            comms.append(2. * sps_matmul(lhs, rhs).apart)
    for lhs, rhs in [(o1h, o2a), (o1a, o2h)]:
        if lhs.num_terms * rhs.num_terms != 0:
            comms.append(2. * sps_matmul(lhs, rhs).hpart)
    return sps_add(*comms, normalize=normalize)


def sps_dot(o1: SparsePauliSum, o2: SparsePauliSum) -> complex:
    """Inner product between two SparsePauliSums."""
    common_entries = np.nonzero(o1.indices[:, None] - o2.indices[None, :] == 0)
    return np.sum(o1.coeffs[common_entries[0]].conjugate() * o2.coeffs[common_entries[1]])
