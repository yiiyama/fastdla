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
def _compute_xmatrix(basis_indices, basis_coeffs, basis_indptr):
    size = len(basis_indptr) - 1
    xmat = np.eye(size, dtype=np.complex128) * 0.5
    for ib1 in range(size):
        start1 = basis_indptr[ib1]
        end1 = basis_indptr[ib1 + 1]
        for ib2 in range(ib1):
            start2 = basis_indptr[ib2]
            end2 = basis_indptr[ib2 + 1]
            xmat[ib1, ib2] = _spv_innerprod_fast(basis_indices[start1:end1],
                                                 basis_coeffs[start1:end1],
                                                 basis_indices[start2:end2],
                                                 basis_coeffs[start2:end2])

    xmat += xmat.T.conjugate()
    return xmat


def compute_xmatrix(basis: Sequence[SparsePauliVector]) -> np.ndarray:
    basis_indices = np.concatenate([op.indices for op in basis])
    basis_coeffs = np.concatenate([op.coeffs for op in basis])
    basis_indptr = np.cumsum([0] + [op.num_terms for op in basis])
    return _compute_xmatrix(basis_indices, basis_coeffs, basis_indptr)


@njit
def _extend_xmatrix(xmat, new_indices, new_coeffs, basis_indices, basis_coeffs, basis_indptr):
    current_size = xmat.shape[0]
    new_xmat = np.empty((current_size + 1, current_size + 1), dtype=np.complex128)
    new_xmat[:current_size, :current_size] = xmat
    for ib in range(current_size):
        start = basis_indptr[ib]
        end = basis_indptr[ib + 1]
        ip = _spv_innerprod_fast(basis_indices[start:end], basis_coeffs[start:end], new_indices,
                                 new_coeffs)
        new_xmat[ib, -1] = ip
        new_xmat[-1, ib] = ip.conjugate()
    new_xmat[-1, -1] = 1.
    return new_xmat


def extend_xmatrix(
    xmat: np.ndarray,
    new_op: SparsePauliVector,
    basis: Sequence[SparsePauliVector]
) -> np.ndarray:
    basis_indices = np.concatenate([op.indices for op in basis])
    basis_coeffs = np.concatenate([op.coeffs for op in basis])
    basis_indptr = np.cumsum([0] + [op.num_terms for op in basis])
    return _extend_xmatrix(xmat, new_op.indices, new_op.coeffs, basis_indices, basis_coeffs,
                           basis_indptr)


@njit
def _is_independent(new_indices, new_coeffs, basis_indices, basis_coeffs, basis_indptr, xmat_inv):
    basis_size = len(basis_indptr) - 1
    pidag_q = np.empty(basis_size, dtype=np.complex128)
    is_zero = True
    for ib in range(basis_size):
        start = basis_indptr[ib]
        end = basis_indptr[ib + 1]
        ip = _spv_innerprod_fast(basis_indices[start:end], basis_coeffs[start:end], new_indices,
                                 new_coeffs)
        pidag_q[ib] = ip
        is_zero &= np.isclose(ip.real, 0.) and np.isclose(ip.imag, 0.)

    if is_zero:
        # Q is orthogonal to all basis vectors
        return True

    # Residual calculation: subtract Pi*ai from Q directly
    a_proj = xmat_inv @ pidag_q
    for ib in range(basis_size):
        a_val = a_proj[ib]
        if np.isclose(a_val.real, 0.) and np.isclose(a_val.imag, 0.):
            continue
        start = basis_indptr[ib]
        end = basis_indptr[ib + 1]
        new_indices, new_coeffs = _spv_sum_fast(new_indices, new_coeffs,
                                                basis_indices[start:end],
                                                -a_val * basis_coeffs[start:end])

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

    basis_indices = np.concatenate([op.indices for op in basis])
    basis_coeffs = np.concatenate([op.coeffs for op in basis])
    basis_indptr = np.cumsum([0] + [op.num_terms for op in basis])
    return _is_independent(new_op.indices, new_op.coeffs, basis_indices, basis_coeffs, basis_indptr,
                           xmat_inv)


@njit
def _main_loop(
    result_indices,
    result_coeffs,
    basis_indices,
    basis_coeffs,
    basis_indptr,
    xmat,
    verbosity
):
    num_results = len(result_indices)
    xmat_inv = np.linalg.inv(xmat)
    prev_indices = np.array([-1], dtype=np.uint64)
    prev_coeffs_conj = None

    for ires in range(num_results):
        if verbosity > 2 and ires % 500 == 0:
            dla_dim = len(basis_indptr) - 1
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

        if _is_independent(indices, coeffs, basis_indices, basis_coeffs, basis_indptr, xmat_inv):
            xmat = _extend_xmatrix(xmat, indices, coeffs, basis_indices, basis_coeffs, basis_indptr)
            xmat_inv = np.linalg.inv(xmat)
            basis_indptr = np.append(basis_indptr, basis_indptr[-1] + indices.shape[0])
            basis_indices = np.concatenate((basis_indices, indices))
            basis_coeffs = np.concatenate((basis_coeffs, coeffs))

    return basis_indices, basis_coeffs, basis_indptr, xmat


def generate_dla(
    generators: Sequence[SparsePauliVector],
    *,
    max_dim: Optional[int] = None,
    min_tasks: int = 0,
    max_workers: Optional[int] = None,
    verbosity: int = 0
) -> list[SparsePauliVector]:
    basis_indices = np.concatenate([op.indices for op in generators])
    basis_coeffs = np.concatenate([op.coeffs / np.sqrt(np.sum(np.square(np.abs(op.coeffs))))
                                   for op in generators])
    basis_indptr = np.cumsum([0] + [op.num_terms for op in generators])
    num_qubits = generators[0].num_qubits

    xmat = _compute_xmatrix(basis_indices, basis_coeffs, basis_indptr)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        commutators = {
            executor.submit(
                _spv_commutator_norm,
                basis_indices[start1:end1],
                basis_coeffs[start1:end1],
                basis_indices[start2:end2],
                basis_coeffs[start2:end2],
                num_qubits
            )
            for i1, (start1, end1) in enumerate(zip(basis_indptr[:-1], basis_indptr[1:]))
            for start2, end2 in zip(basis_indptr[:i1], basis_indptr[1:i1 + 1])
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
            old_dim = basis_indptr.shape[0] - 1
            basis_indices, basis_coeffs, basis_indptr, xmat = _main_loop(
                result_indices, result_coeffs, basis_indices, basis_coeffs, basis_indptr, xmat,
                verbosity
            )
            new_dim = basis_indptr.shape[0] - 1
            if verbosity > 2:
                print(f'Found {new_dim - old_dim} new ops in {time.time() - main_loop_start:.2f}s')

            new_commutators = []
            for ib1 in range(old_dim, new_dim):
                start1, end1 = basis_indptr[ib1:ib1 + 2]
                for ib2 in range(ib1):
                    start2, end2 = basis_indptr[ib2:ib2 + 2]
                    new_commutators.append(
                        executor.submit(
                            _spv_commutator_norm,
                            basis_indices[start1:end1],
                            basis_coeffs[start1:end1],
                            basis_indices[start2:end2],
                            basis_coeffs[start2:end2],
                            num_qubits
                        )
                    )
            commutators.update(new_commutators)
            if verbosity > 1 and new_commutators:
                print(f'Adding {len(new_commutators)} commutators; total {len(commutators)}')
            if verbosity > 0:
                print(f'Current DLA dimension: {new_dim}')
            if verbosity > 2:
                print(f'Outer loop took {time.time() - outer_loop_start:.2f}s')

            if max_dim is not None and new_dim >= max_dim:
                basis_indptr = basis_indptr[:max_dim + 1]
                basis_indices = basis_indices[:basis_indptr[-1]]
                basis_coeffs = basis_coeffs[:basis_indptr[-1]]
                executor.shutdown(wait=False, cancel_futures=True)
                break

            while True:
                done, not_done = wait(commutators, return_when=FIRST_COMPLETED)
                if len(not_done) == 0 or len(done) > min_tasks:
                    break
                time.sleep(1.)

    return [
        SparsePauliVector(
            basis_indices[start:end],
            basis_coeffs[start:end],
            num_qubits=num_qubits,
            no_check=True
        )
        for start, end in zip(basis_indptr[:-1], basis_indptr[1:])
    ]
