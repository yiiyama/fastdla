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
])
PAULI_PROD_COEFF = np.array([
    [1., 1., 1., 1.],
    [1., 1., 1.j, -1.j],
    [1., -1.j, 1., 1.j],
    [1., 1.j, -1.j, 1.]
])


class SparsePauliOp:
    """Purpose-specific implementation of a sparse Pauli operator."""
    @staticmethod
    def str_to_idx(pstr: str) -> int:
        return np.ravel_multi_index(tuple(PAULI_INDICES[p] for p in pstr),
                                    (4,) * len(pstr))

    @staticmethod
    def idx_to_str(idx: int, num_qubits: int) -> str:
        return ''.join(PAULI_NAMES[i] for i in np.unravel_index(idx, (4,) * num_qubits))

    def __init__(
        self,
        strings: Union[str, int, list[str], list[int]],
        coeffs: Union[Number, list[Number]],
        num_qubits=None
    ):
        if isinstance(strings, (str, int)):
            strings = [strings]
        if isinstance(coeffs, Number):
            coeffs = [coeffs]

        if len(strings) == 0:
            self.num_qubits = num_qubits
            self.strings = np.array([])
            self.coeffs = np.array([])
            return

        if isinstance(strings[0], str):
            num_qubits = len(strings[0])
            strings = [self.str_to_idx(pstr) for pstr in strings]
        elif num_qubits is None:
            raise ValueError('Need num_qubits')

        self.num_qubits = num_qubits
        indices = np.argsort(strings)
        self.strings = np.asarray(strings)[indices]
        self.coeffs = np.asarray(coeffs, dtype=complex)[indices]

        if self.strings.shape != self.coeffs.shape:
            raise ValueError('Strings and coeffs shape mismatch')

    def __str__(self) -> str:
        opstr = ', '.join(f'{coeff} {self.idx_to_str(idx, self.num_qubits)}'
                          for coeff, idx in zip(self.coeffs, self.strings))
        return f'[{opstr}]'

    def __repr__(self) -> str:
        return ('SparsePauliOp(['
                + (', '.join(f"'{self.idx_to_str(idx, self.num_qubits)}'" for idx in self.strings))
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
        return self.strings.shape[0]

    @property
    def hpart(self) -> 'SparsePauliOp':
        """Return a new SPO with Hermitian terms only."""
        indices = np.nonzero(np.logical_not(np.isclose(self.coeffs.real, 0.)))[0]
        return SparsePauliOp(self.strings[indices], self.coeffs[indices].real, self.num_qubits)

    @property
    def apart(self) -> 'SparsePauliOp':
        """Return a new SPO with anti-Hermitian terms only."""
        indices = np.nonzero(np.logical_not(np.isclose(self.coeffs.imag, 0.)))[0]
        return SparsePauliOp(self.strings[indices], 1.j * self.coeffs[indices].imag,
                             self.num_qubits)

    def __add__(self, other: 'SparsePauliOp') -> 'SparsePauliOp':
        return spo_sum(self, other)

    def __mul__(self, scalar: Number) -> 'SparsePauliOp':
        try:
            coeffs = scalar * self.coeffs
        except Exception:
            return NotImplemented
        return SparsePauliOp(self.strings, coeffs, self.num_qubits)

    def __rmul__(self, scalar: Number) -> 'SparsePauliOp':
        return self.__mul__(scalar)

    def __matmul__(self, other: 'SparsePauliOp') -> 'SparsePauliOp':
        return spo_prod(self, other)

    def commutator(self, other: 'SparsePauliOp') -> 'SparsePauliOp':
        return spo_commutator(self, other)

    def to_csc(self) -> csc_array:
        shape = (self.vlen, 1)
        return csc_array((self.coeffs, self.strings, [0, self.num_terms]), shape=shape)

    @staticmethod
    def from_csc(array: csc_array, num_qubits: int) -> 'SparsePauliOp':
        return SparsePauliOp(array.indices, array.data, num_qubits)


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


def spo_sum(*ops) -> SparsePauliOp:
    """Sum of two sparse Pauli ops."""
    strings, coeffs = _uniquify(
        np.concatenate([op.strings for op in ops]),
        np.concatenate([op.coeffs for op in ops])
    )
    return SparsePauliOp(strings, coeffs, ops[0].num_qubits)


def spo_prod(o1: SparsePauliOp, o2: SparsePauliOp) -> SparsePauliOp:
    """Product of two sparse Pauli ops."""
    if o1.num_qubits != o2.num_qubits:
        raise ValueError('Matmul between incompatible SparsePauliOps')

    if (num_terms := o1.num_terms * o2.num_terms) == 0:
        return SparsePauliOp([], [], o1.num_qubits)

    indices = np.unravel_index(np.arange(num_terms), (o1.num_terms, o2.num_terms))
    coeffs = np.outer(o1.coeffs, o2.coeffs).reshape(-1)
    s1s = o1.strings[indices[0]]
    s2s = o2.strings[indices[1]]

    shape = o1.shape
    p1s = np.array(np.unravel_index(s1s, shape=shape))  # shape [num_qubits, num_strings]
    p2s = np.array(np.unravel_index(s2s, shape=shape))
    pout = PAULI_PROD_INDEX[p1s, p2s]
    sout = np.ravel_multi_index(tuple(pout), shape)  # shape [num_strings]
    coeffs *= np.prod(PAULI_PROD_COEFF[p1s, p2s], axis=0)

    strings, coeffs = _uniquify(sout, coeffs)
    return SparsePauliOp(strings, coeffs, o1.num_qubits)


def spo_commutator(o1: SparsePauliOp, o2: SparsePauliOp) -> SparsePauliOp:
    o1h = o1.hpart
    o1a = o1.apart
    o2h = o2.hpart
    o2a = o2.apart
    comms = []
    for lhs, rhs in [(o1h, o2h), (o1a, o2a)]:
        if lhs.num_terms * rhs.num_terms != 0:
            comms.append(2. * spo_prod(lhs, rhs).apart)
    for lhs, rhs in [(o1h, o2a), (o1a, o2h)]:
        if lhs.num_terms * rhs.num_terms != 0:
            comms.append(2. * spo_prod(lhs, rhs).hpart)
    return spo_sum(*comms)


def spolist_to_csc(ops: list[SparsePauliOp]) -> csc_array:
    if len(ops) == 0:
        raise ValueError('Cannot convert zero-length list to CSC')

    shape = (ops[0].vlen, len(ops))
    data = np.concatenate([op.coeffs for op in ops])
    indices = np.concatenate([op.strings for op in ops])
    indptr = np.cumsum([0] + [op.num_terms for op in ops])
    return csc_array((data, indices, indptr), shape=shape)


def csc_to_spolist(matrix: csc_array) -> list[SparsePauliOp]:
    num_qubits = int(np.round(np.emath.logn(4, matrix.shape[0])))
    ops = []
    for icol in range(matrix.shape[1]):
        idx_begin, idx_end = matrix.indptr[icol:icol+2]
        ops.append(SparsePauliOp(matrix.indices[idx_begin:idx_end],
                                 matrix.data[idx_begin:idx_end],
                                 num_qubits=num_qubits))

    return ops


def _check_linear_independence(
    basis_matrix: csc_array,
    xmat_inv: np.ndarray,
    new_op: csc_array
) -> bool:
    """
    Let the known dla basis ops be P0, P1, ..., Pn. The basis_matrix Π is a matrix formed by
    stacking the column vectors {Pi}:
    Π = (P0, P1, ..., Pn).
    If test_op Q is linearly dependent on {Pi}, there is a column vector of coefficients
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
    pidag_q = new_op.conjugate().transpose(copy=False).dot(basis_matrix).toarray().conjugate().T
    if np.allclose(pidag_q, 0.):
        # Q is orthogonal to all basis vectors
        return True
    a_proj = xmat_inv @ pidag_q
    residual = new_op - basis_matrix.dot(a_proj)
    return not np.allclose(residual.data, 0.)


def _update_basis_matrix(
    basis_matrix: csc_array,
    # xmat: np.ndarray,
    new_op: csc_array
) -> csc_array:
    """Extend the basis matrix by one row and column using new_op."""
    # # Adding the nth row & col to X
    # # New row (length n-1) of X: xn† = Pn† Π
    # new_op_dag = new_op.conjugate()
    # xmat_new_row = new_op_dag.dot(basis_matrix).toarray()
    # # New diagonal element of X: Pn† Pn
    # op_mod2 = np.array([new_op_dag.data @ new_op.data])
    # # New col (full length) of X: (xn, |Pn|^2)
    # xmat_new_col = np.concatenate([xmat_new_row.conjugate(), op_mod2])
    # # Expand X
    # xmat = np.concatenate([xmat, xmat_new_row.reshape(1, -1)], axis=0)
    # xmat = np.concatenate([xmat, xmat_new_col.reshape(-1, 1)], axis=1)

    old_num_basis = basis_matrix.shape[1]
    data = np.concatenate([basis_matrix.data, new_op.data])
    indices = np.concatenate([basis_matrix.indices, new_op.indices])
    indptr = np.concatenate([basis_matrix.indptr, [basis_matrix.indptr[-1] + new_op.nnz]])
    new_shape = (basis_matrix.shape[0], old_num_basis + 1)
    basis_matrix = csc_array((data, indices, indptr), shape=new_shape)

    return basis_matrix


def _extend_dla_basis(
    basis_matrix: csc_array,
    start_idx: int
) -> csc_array:
    """Compute the commutators among the basis operators and identify linearly independent outcomes.
    """
    xmat = basis_matrix.conjugate().transpose(copy=False).dot(basis_matrix).toarray()
    xmat_inv = np.linalg.inv(xmat)

    num_qubits = int(np.round(np.emath.logn(4, basis_matrix.shape[0])))

    for i1 in range(start_idx, basis_matrix.shape[1]):
        # Construct a SparsePauliOp from the i1th column of basis_matrix
        idx_begin, idx_end = basis_matrix.indptr[i1:i1+2]
        g1 = SparsePauliOp(basis_matrix.indices[idx_begin:idx_end],
                           basis_matrix.data[idx_begin:idx_end],
                           num_qubits=num_qubits)
        # Compute the commutators up to column i1 - 1
        for i2 in range(i1):
            # Construct a SparsePauliOp from the i2th column of basis_matrix
            idx_begin, idx_end = basis_matrix.indptr[i2:i2+2]
            g2 = SparsePauliOp(basis_matrix.indices[idx_begin:idx_end],
                               basis_matrix.data[idx_begin:idx_end],
                               num_qubits=num_qubits)
            new_op = spo_commutator(g1, g2)
            if new_op.strings.shape[0] == 0:
                # [g1, g2] = 0
                continue

            new_op_csc = new_op.to_csc()

            if not _check_linear_independence(basis_matrix, xmat_inv, new_op_csc):
                continue

            # op is linearly independent -> Append to basis_matrix
            basis_matrix = _update_basis_matrix(basis_matrix, new_op_csc)
            xmat = basis_matrix.conjugate().transpose(copy=False).dot(basis_matrix).toarray()
            xmat_inv = np.linalg.inv(xmat)

    return basis_matrix


def full_dla_basis(generators: list[SparsePauliOp]) -> list[SparsePauliOp]:
    basis_matrix = spolist_to_csc(generators)
    start_idx = 0
    iteration = 0
    while True:
        print('iteration', iteration, 'num_basis', start_idx)
        old_num_basis = basis_matrix.shape[1]
        basis_matrix = _extend_dla_basis(basis_matrix, start_idx)
        if basis_matrix.shape[1] == old_num_basis:
            # No new basis added
            break

        start_idx = old_num_basis

    return csc_to_spolist(basis_matrix)
