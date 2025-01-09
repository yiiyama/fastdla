"""Calculation of DLA based on dense matrices."""
from collections.abc import Sequence
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Optional
import time
import numpy as np
from numba import njit, objmode


@njit
def _commutator_norm(
    op1: np.ndarray,
    op2: np.ndarray
) -> np.ndarray:
    comm = op1 @ op2 - op2 @ op1
    norm = np.sqrt(np.abs(np.trace(comm.conjugate().T @ comm)))
    if np.isclose(norm, 0.):
        return np.zeros_like(op1)
    else:
        return comm / norm


def compute_xmatrix(basis: Sequence[np.ndarray]) -> np.ndarray:
    basis = np.asarray(basis)
    return np.einsum('ijk,ljk->il', basis.conjugate(), basis)


@njit
def _extend_xmatrix(xmat, new_op, basis):
    current_size = xmat.shape[0]
    new_xmat = np.empty((current_size + 1, current_size + 1), dtype=np.complex128)
    new_xmat[:current_size, :current_size] = xmat
    for ib in range(current_size):
        ip = np.trace(basis[ib].conjugate().T @ new_op)
        new_xmat[ib, -1] = ip
        new_xmat[-1, ib] = ip.conjugate()
    new_xmat[-1, -1] = 1.
    return new_xmat


def extend_xmatrix(
    xmat: np.ndarray,
    new_op: np.ndarray,
    basis: Sequence[np.ndarray]
) -> np.ndarray:
    return _extend_xmatrix(xmat, new_op, np.asarray(basis))


@njit
def _is_independent(new_op, basis, xmat_inv):
    basis_size = basis.shape[0]
    pidag_q = np.empty(basis_size, dtype=np.complex128)
    is_zero = True
    for ib in range(basis_size):
        ip = np.trace(basis[ib].conjugate().T @ new_op)
        pidag_q[ib] = ip
        is_zero &= np.isclose(ip.real, 0.) and np.isclose(ip.imag, 0.)

    if is_zero:
        # Q is orthogonal to all basis vectors
        return True

    # Residual calculation: subtract Pi*ai from Q directly
    a_proj = xmat_inv @ pidag_q
    new_op -= np.sum(basis * a_proj[:, None, None], axis=0)
    return not (np.allclose(new_op.real, 0.) and np.allclose(new_op.imag, 0.))


def is_independent(
    new_op: np.ndarray,
    basis: Sequence[np.ndarray],
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

    return _is_independent(new_op, np.asarray(basis), xmat_inv)


@njit
def _main_loop(results, basis, xmat, verbosity):
    num_results = results.shape[0]
    xmat_inv = np.linalg.inv(xmat)

    for ires in range(num_results):
        if verbosity > 2 and ires % 500 == 0:
            dla_dim = basis.shape[0]
            with objmode():
                print(f'Processing SPV {ires}/{num_results} (DLA dim {dla_dim})..', flush=True)

        comm = results[ires]
        if np.allclose(comm.real, 0.) and np.allclose(comm.imag, 0.):
            continue

        if _is_independent(comm, basis, xmat_inv):
            xmat = _extend_xmatrix(xmat, comm, basis)
            xmat_inv = np.linalg.inv(xmat)
            basis = np.concatenate((basis, comm[None, :, :]), axis=0)

    return basis, xmat


def generate_dla(
    generators: Sequence[np.ndarray],
    *,
    max_dim: Optional[int] = None,
    min_tasks: int = 0,
    max_workers: Optional[int] = None,
    verbosity: int = 0
) -> list[np.ndarray]:
    basis = np.asarray(generators)
    xmat = compute_xmatrix(basis)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        commutators = {
            executor.submit(_commutator_norm, op1, op2)
            for i1, op1 in enumerate(basis)
            for op2 in basis[:i1]
        }

        if verbosity > 1:
            print(f'Starting with {len(commutators)} commutators..')

        done, _ = wait(commutators, return_when=ALL_COMPLETED)

        while commutators:
            outer_loop_start = time.time()
            if verbosity > 1:
                print(f'Evaluating {len(done)}/{len(commutators)} commutators for independence')

            commutators.difference_update(done)
            results = np.array([fut.result() for fut in done])

            main_loop_start = time.time()
            old_dim = basis.shape[0]
            basis, xmat = _main_loop(results, basis, xmat, verbosity)
            new_dim = basis.shape[0]
            if verbosity > 2:
                print(f'Found {new_dim - old_dim} new ops in {time.time() - main_loop_start:.2f}s')

            new_commutators = [
                executor.submit(_commutator_norm, basis[ib1], basis[ib2])
                for ib1 in range(old_dim, new_dim)
                for ib2 in range(ib1)
            ]
            commutators.update(new_commutators)
            if verbosity > 1 and new_commutators:
                print(f'Adding {len(new_commutators)} commutators; total {len(commutators)}')
            if verbosity > 0:
                print(f'Current DLA dimension: {new_dim}')
            if verbosity > 2:
                print(f'Outer loop took {time.time() - outer_loop_start:.2f}s')

            if max_dim is not None and new_dim >= max_dim:
                basis = basis[:max_dim]
                executor.shutdown(wait=False, cancel_futures=True)
                break

            while True:
                done, not_done = wait(commutators, return_when=FIRST_COMPLETED)
                if len(not_done) == 0 or len(done) > min_tasks:
                    break
                time.sleep(1.)

    return list(basis)
