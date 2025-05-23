"""Pauli matrices and tensor products thereof."""
from collections.abc import Sequence
from typing import Optional
import numpy as np
from scipy.sparse import csr_array


PAULI_NAMES = ['I', 'X', 'Y', 'Z']
PAULI_INDICES = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
PAULI_MULT_INDEX = np.array([
    [0, 1, 2, 3],
    [1, 0, 3, 2],
    [2, 3, 0, 1],
    [3, 2, 1, 0]
], dtype=np.uint64)
PAULI_MULT_COEFF = np.array([
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
PAULI_SPARSE_COLS = np.array([
    [0, 1],
    [1, 0],
    [1, 0],
    [0, 1]
])
PAULI_SPARSE_DATA = np.array([
    [1., 1.],
    [1., 1.],
    [-1.j, 1.j],
    [1., -1.]
])


class PauliProduct:
    """Product of Pauli operators."""
    @staticmethod
    def str_to_idx(pstr: str) -> int:
        indices = np.array([PAULI_INDICES[p.upper()] for p in pstr], dtype=np.uint64)
        return np.sum(indices * (4 ** np.arange(indices.shape[0])[::-1]))

    @staticmethod
    def seq_to_idx(arr: Sequence[int]) -> int:
        indices = np.asarray(arr)
        if np.any(np.logical_or(indices < 0, indices > 3)):
            raise ValueError('Invalid Pauli index')
        return np.sum(indices * (4 ** np.arange(indices.shape[0])))

    @staticmethod
    def matmul(
        index1: int | np.ndarray,
        coeff1: complex | np.ndarray,
        index2: int | np.ndarray,
        coeff2: complex | np.ndarray,
        num_qubits: int
    ) -> tuple[int | np.ndarray, complex | np.ndarray]:
        index1 = np.asarray(index1)
        coeff1 = np.asarray(coeff1)
        index2 = np.asarray(index2)
        coeff2 = np.asarray(coeff2)
        powers = np.expand_dims(4 ** np.arange(num_qubits)[::-1], tuple(range(len(index1.shape))))

        p1 = ((np.expand_dims(index1, -1) // powers) % 4).astype(int)  # [..., num_qubits]
        p2 = ((np.expand_dims(index2, -1) // powers) % 4).astype(int)  # [..., num_qubits]
        pout = PAULI_MULT_INDEX[p1, p2]  # [..., num_qubits]
        idxout = np.sum(pout * powers, axis=-1)
        coeffout = coeff1 * coeff2 * np.prod(PAULI_MULT_COEFF[p1, p2], axis=-1)

        return idxout, coeffout

    def __init__(
        self,
        index: str | int | Sequence[int],
        num_qubits: Optional[int] = None
    ):
        try:
            num_qubits = len(index)
        except TypeError as exc:
            if num_qubits is None:
                raise ValueError('Need num_qubits if index is an int') from exc
            self.index = index
        else:
            if num_qubits == 0:
                raise ValueError('Null sequence')
            if isinstance(index, str):
                try:
                    self.index = PauliProduct.str_to_idx(index)
                except KeyError as ex:
                    raise ValueError('Invalid Pauli string') from ex
            else:
                try:
                    self.index = PauliProduct.seq_to_idx(index)
                except ValueError as ex:
                    raise ValueError('Invalid Pauli index') from ex

        self.num_qubits = num_qubits

    def __str__(self) -> str:
        return ''.join(PAULI_NAMES[i] for i in self.indices()[::-1])

    def __repr__(self) -> str:
        return f"PauliProduct('{str(self)}')"

    def indices(self) -> np.ndarray:
        return ((self.index // (4 ** np.arange(self.num_qubits))) % 4).astype(int)

    def to_matrix(self, *, sparse: bool = False, npmod=np) -> np.ndarray | csr_array:
        if sparse:
            cols = npmod.array(0)
            data = npmod.array(1.)
            for iq, ip in enumerate(self.indices()):
                cols = npmod.add.outer(PAULI_SPARSE_COLS[ip] * (2 ** iq), cols).reshape(-1)
                data = npmod.outer(PAULI_SPARSE_DATA[ip], data).reshape(-1)
            dim = 2 ** self.num_qubits
            indptr = npmod.arange(dim + 1)
            if npmod is np:
                return csr_array((data, cols, indptr), shape=(dim, dim))
            else:
                # pylint: disable-next=import-outside-toplevel
                from jax.experimental.sparse import BCSR
                return BCSR((data, cols, indptr), shape=(dim, dim), indices_sorted=True,
                            unique_indices=True)

        matrix = np.array(1.)
        for ip in self.indices():
            matrix = np.kron(PAULIS[ip], matrix)
        return matrix
