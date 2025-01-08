"""Calculation of DLA."""
from collections.abc import Sequence
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Optional
import time
import numpy as np
from numba import njit, objmode
from .sparse_pauli_vector import SparsePauliVector
from .spv_fast import _spv_commutator_fast, _spv_innerprod_fast, _spv_sum_fast


@njit(nogil=True)
def _spv_commutator_norm(
    indices1: np.ndarray,
    coeffs1: np.ndarray,
    indices2: np.ndarray,
    coeffs2: np.ndarray,
    num_qubits: int
) -> tuple[np.ndarray, np.ndarray]:
    indices, coeffs = _spv_commutator_fast(indices1, coeffs1, indices2, coeffs2, num_qubits)
    return indices, coeffs / np.sqrt(np.sum(np.square(np.abs(coeffs))))


@njit
def _compute_xmatrix(basis_indices, basis_coeffs):
    size = len(basis_indices)
    xmat = np.eye(size, dtype=np.complex128) * 0.5
    for i1, (indices1, coeffs1) in enumerate(zip(basis_indices, basis_coeffs)):
        for i2, (indices2, coeffs2) in enumerate(zip(basis_indices[:i1], basis_coeffs[:i1])):
            xmat[i1, i2] = _spv_innerprod_fast(indices1, coeffs1, indices2, coeffs2)

    xmat += xmat.T.conjugate()
    return xmat


def compute_xmatrix(basis: Sequence[SparsePauliVector]) -> np.ndarray:
    return _compute_xmatrix([op.indices for op in basis], [op.coeffs for op in basis])


@njit
def _extend_xmatrix(xmat, basis_indices, basis_coeffs, new_indices, new_coeffs):
    current_size = xmat.shape[0]
    new_xmat = np.empty((current_size + 1, current_size + 1), dtype=np.complex128)
    new_xmat[:current_size, :current_size] = xmat
    for ib in range(current_size):
        ip = _spv_innerprod_fast(basis_indices[ib], basis_coeffs[ib], new_indices, new_coeffs)
        new_xmat[ib, -1] = ip
        new_xmat[-1, ib] = ip.conjugate()
    new_xmat[-1, -1] = 1.
    return new_xmat


def extend_xmatrix(
    xmat: np.ndarray,
    basis: Sequence[SparsePauliVector],
    new_op: SparsePauliVector
) -> np.ndarray:
    return _extend_xmatrix(xmat, [op.indices for op in basis], [op.coeffs for op in basis],
                           new_op.indices, new_op.coeffs)


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

    # Residual calculation: subtract Pi*ai from Q directly
    a_proj = xmat_inv @ pidag_q
    for a_val, bindices, bcoeffs in zip(a_proj, basis_indices, basis_coeffs):
        if np.isclose(a_val.real, 0.) and np.isclose(a_val.imag, 0.):
            continue
        new_indices, new_coeffs = _spv_sum_fast(new_indices, new_coeffs,
                                                bindices, -a_val * bcoeffs)

    return new_indices.shape[0] != 0


def is_independent(
    new_op: SparsePauliVector,
    basis: list[SparsePauliVector],
    xmat_inv: Optional[np.ndarray] = None,
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
    if xmat_inv is None:
        xmat_inv = np.linalg.inv(compute_xmatrix(basis))

    return _is_independent(new_op.indices, new_op.coeffs,
                           [op.indices for op in basis], [op.coeffs for op in basis],
                           xmat_inv)


@njit
def _main_loop(basis_indices, basis_coeffs, xmat_ref, result_indices, result_coeffs, verbosity):
    num_results = len(result_indices)
    new_ops = []
    xmat = xmat_ref[0]
    xmat_inv = np.linalg.inv(xmat)
    prev_indices = np.array([-1], dtype=np.uint64)
    prev_coeffs_conj = None

    for ires in range(num_results):
        if verbosity > 2 and ires % 100 == 0:
            dla_dim = len(basis_indices)
            with objmode():
                print(f'Processing SPV {ires}/{num_results} (DLA dim {dla_dim})..', flush=True)

        indices = result_indices[ires]
        if indices.shape[0] == 0:
            continue

        coeffs = result_coeffs[ires]
        if (prev_indices.shape[0] == indices.shape[0] and np.all(prev_indices == indices)
                and np.isclose(np.abs(prev_coeffs_conj @ coeffs), 1.)):
            continue

        prev_indices = indices
        prev_coeffs_conj = coeffs.conjugate()

        if _is_independent(indices, coeffs, basis_indices, basis_coeffs, xmat_inv):
            new_ops.append((indices, coeffs, len(basis_indices)))
            xmat = _extend_xmatrix(xmat, basis_indices, basis_coeffs, indices, coeffs)
            xmat_inv = np.linalg.inv(xmat)
            basis_indices.append(indices)
            basis_coeffs.append(coeffs)

    xmat_ref[0] = xmat
    return new_ops


def generate_dla(
    generators: Sequence[SparsePauliVector],
    *,
    max_workers: Optional[int] = None,
    verbosity: int = 0
) -> list[SparsePauliVector]:
    basis_indices = [op.indices for op in generators]
    basis_coeffs = [op.coeffs / np.sqrt(np.sum(np.square(np.abs(op.coeffs)))) for op in generators]
    num_qubits = generators[0].num_qubits

    xmat_ref = [_compute_xmatrix(basis_indices, basis_coeffs)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        commutators = {
            executor.submit(_spv_commutator_norm, idx1, c1, idx2, c2, num_qubits)
            for i1, (idx1, c1) in enumerate(zip(basis_indices, basis_coeffs))
            for idx2, c2 in zip(basis_indices[:i1], basis_coeffs[:i1])
        }

        if verbosity > 1:
            print(f'Starting with {len(commutators)} commutators..')

        done, _ = wait(commutators, return_when=ALL_COMPLETED)

        while commutators:
            outer_loop_start = time.time()
            if verbosity > 1:
                print(f'Evaluating {len(done)}/{len(commutators)} commutators for independence')

            commutators.difference_update(done)
            results = [fut.result() for fut in done]

            sort_start = time.time()
            indices_tuples = [tuple(res[0]) for res in results]
            sort_idx = sorted(range(len(results)), key=indices_tuples.__getitem__)
            result_indices = [results[idx][0] for idx in sort_idx]
            result_coeffs = [results[idx][1] for idx in sort_idx]
            if verbosity > 2:
                print(f'Sorted in {time.time() - sort_start:.2f}s')

            main_loop_start = time.time()
            new_ops = _main_loop(basis_indices, basis_coeffs, xmat_ref, result_indices,
                                 result_coeffs, verbosity)
            if verbosity > 2:
                print(f'Found {len(new_ops)} new ops in {time.time() - main_loop_start:.2f}s')

            new_commutators = [
                executor.submit(_spv_commutator_norm, indices, coeffs, idx, c, num_qubits)
                for indices, coeffs, basis_upto in new_ops
                for idx, c in zip(basis_indices[:basis_upto], basis_coeffs[:basis_upto])
            ]
            commutators.update(new_commutators)
            if verbosity > 1:
                print(f'Adding {len(new_commutators)} commutators; total {len(commutators)}')
            if new_ops and verbosity > 0:
                print(f'Current DLA dimension: {len(basis_indices)}')
            if verbosity > 2:
                print(f'Outer loop took {time.time() - outer_loop_start:.2f}s')

            done, _ = wait(commutators, return_when=FIRST_COMPLETED)

    return [SparsePauliVector(idx, c, num_qubits=num_qubits, no_check=True)
            for idx, c in zip(basis_indices, basis_coeffs)]
