"""Implementation of the Lie closure generator using SparsePauliVectors."""
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import logging
from typing import Any, Optional
import time
from multiprocessing import cpu_count
import numpy as np
from numba import njit, objmode
from fastdla.sparse_pauli_vector import SparsePauliVector, SparsePauliVectorArray
from fastdla.spv_fast import _uniquify_fast, _spv_commutator_fast, _spv_dot_fast

LOG = logging.getLogger(__name__)
BASIS_ALLOC_UNIT = 1024
MEM_ALLOC_UNIT = SparsePauliVectorArray.MEM_ALLOC_UNIT


@njit
def _linear_independence(
    new_indices: np.ndarray,
    new_coeffs: np.ndarray,
    basis_indices: np.ndarray,
    basis_coeffs: np.ndarray,
    basis_ptrs: list[int],
    xinv: np.ndarray,
    log_level: int
) -> tuple[bool, np.ndarray]:
    """Check linear independence of a new Pauli vector."""
    basis_size = len(basis_ptrs) - 1
    # Create and fill the Π†Q vector
    pidag_q = np.empty(basis_size, dtype=np.complex128)
    is_zero = True
    for ib in range(basis_size):
        start, end = basis_ptrs[ib:ib + 2]
        ip = _spv_dot_fast(basis_indices[start:end], basis_coeffs[start:end],
                           new_indices, new_coeffs)
        pidag_q[ib] = ip
        is_zero &= np.isclose(ip.real, 0.) and np.isclose(ip.imag, 0.)

    if is_zero:
        # Q is orthogonal to all basis vectors
        if log_level <= logging.DEBUG:
            with objmode():
                pauli_sum = []
                for t in zip(new_indices, new_coeffs):
                    pauli_sum.append(str(t))
                LOG.debug('%s is orthogonal to all basis vectors', ','.join(pauli_sum))

        return True, pidag_q

    # Residual calculation: uniquify the concatenation of Q and columns of -Pi*ai
    concat_size = new_indices.shape[0]
    nonzero_idx = []
    avals = []
    for ib in range(basis_size):
        # a = X^{-1}Π†Q
        # aval = xinv[ib * basis_size:(ib + 1) * basis_size] @ pidag_q
        aval = xinv[ib] @ pidag_q
        if not (np.isclose(aval.real, 0.) and np.isclose(aval.imag, 0.)):
            nonzero_idx.append(ib)
            avals.append(aval)
            concat_size += basis_ptrs[ib + 1] - basis_ptrs[ib]

    concat_indices = np.empty(concat_size, dtype=basis_indices.dtype)
    concat_coeffs = np.empty(concat_size, dtype=basis_coeffs.dtype)
    concat_indices[:new_indices.shape[0]] = new_indices
    concat_coeffs[:new_coeffs.shape[0]] = new_coeffs
    current_pos = new_indices.shape[0]
    for ib, aval in zip(nonzero_idx, avals):
        start, end = basis_ptrs[ib:ib + 2]
        next_pos = current_pos + end - start
        concat_indices[current_pos:next_pos] = basis_indices[start:end]
        concat_coeffs[current_pos:next_pos] = -aval * basis_coeffs[start:end]
        current_pos = next_pos

    indices, _ = _uniquify_fast(concat_indices, concat_coeffs, False)
    is_independent = indices.shape[0] != 0

    if log_level <= logging.DEBUG:
        num_overlaps = len(nonzero_idx)
        with objmode():
            if is_independent:
                LOG.debug('New op has overlaps with %d basis vectors but is linearly independent',
                          num_overlaps)
            else:
                LOG.debug('New op exists in the span and overlaps with %d basis vectors',
                          num_overlaps)

    return is_independent, pidag_q


def linear_independence(
    new_op: SparsePauliVector,
    basis: SparsePauliVectorArray,
    xinv: Optional[np.ndarray] = None
) -> bool:
    """Check if the given operator is linearly independent from all other elements in the basis.

    See the docstring of generator.linear_independence for details of the algorithm.

    Args:
        new_op: Lie algebra element Q to check the linear independence of.
        basis: The basis (list of linearly independent elements) of the Lie Algebra.
        xinv: Inverse of the X matrix.

    Returns:
        True if Q is linearly independent from all elements of the basis.
    """
    if xinv is None:
        xmat = np.eye((basis_size := len(basis)), dtype=basis.coeffs.dtype)
        for col in range(1, basis_size):
            col_start, col_end = basis.ptrs[col:col + 2]
            for row in range(col):
                row_start, row_end = basis.ptrs[row:row + 2]
                ip = _spv_dot_fast(
                    basis.indices[row_start:row_end],
                    basis.coeffs[row_start:row_end],
                    basis.indices[col_start:col_end],
                    basis.coeffs[col_start:col_end]
                )
                xmat[row, col] = ip
                xmat[col, row] = ip.conjugate()

        xinv = np.ascontiguousarray(np.linalg.inv(xmat))

    return _linear_independence(new_op.indices, new_op.coeffs,
                                basis.indices, basis.coeffs, basis.ptrs,
                                xinv, LOG.getEffectiveLevel())[0]


@njit
def _orthogonalize(
    new_indices: np.ndarray,
    new_coeffs: np.ndarray,
    basis_indices: np.ndarray,
    basis_coeffs: np.ndarray,
    basis_ptrs: list[int],
    log_level: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the component of a vector with respect to an orthonormal basis."""
    # Strategy: Calculate v - sum_j (v.b_j)b_j through uniquify()
    basis_size = len(basis_ptrs) - 1
    concat_coeffs = np.zeros(basis_ptrs[-1] + new_coeffs.shape[0], dtype=basis_coeffs.dtype)

    for ib in range(basis_size):
        start, end = basis_ptrs[ib:ib + 2]
        ip = _spv_dot_fast(basis_indices[start:end], basis_coeffs[start:end],
                           new_indices, new_coeffs)
        if not (np.isclose(ip.real, 0.) and np.isclose(ip.imag, 0.)):
            concat_coeffs[start:end] = -ip * basis_coeffs[start:end]

    concat_coeffs[basis_ptrs[-1]:] = new_coeffs
    concat_indices = np.concatenate((basis_indices[:basis_ptrs[-1]], new_indices))
    return _uniquify_fast(concat_indices, concat_coeffs, False)


def orthogonalize(
    new_op: SparsePauliVector,
    basis: SparsePauliVectorArray
) -> SparsePauliVector:
    """Subtract the subspace projection of an algebra element from itself."""
    indices, coeffs = _orthogonalize(new_op.indices, new_op.coeffs,
                                     basis.indices, basis.coeffs, basis.ptrs,
                                     LOG.getEffectiveLevel())
    return SparsePauliVector(indices, coeffs, num_qubits=new_op.num_qubits, no_check=True)


@njit
def _update_basis(indices, coeffs, basis_indices, basis_coeffs, basis_ptrs, log_level):
    next_ptr = basis_ptrs[-1] + indices.shape[0]
    if next_ptr > basis_indices.shape[0]:
        # At maximum capacity -> reallocate
        additional_capacity = (((next_ptr - basis_indices.shape[0]) // MEM_ALLOC_UNIT + 1)
                               * MEM_ALLOC_UNIT)
        basis_indices = np.concatenate((
            basis_indices,
            np.empty(additional_capacity, dtype=basis_indices.dtype)
        ))
        basis_coeffs = np.concatenate((
            basis_coeffs,
            np.empty(additional_capacity, dtype=basis_coeffs.dtype)
        ))

    basis_ptrs.append(next_ptr)
    basis_indices[basis_ptrs[-2]:basis_ptrs[-1]] = indices
    basis_coeffs[basis_ptrs[-2]:basis_ptrs[-1]] = coeffs

    return basis_indices, basis_coeffs


@njit
def _if_independent_update(
    indices: np.ndarray,
    coeffs: np.ndarray,
    basis_indices: np.ndarray,
    basis_coeffs: np.ndarray,
    basis_ptrs: np.ndarray,
    aux: tuple[np.ndarray, np.ndarray],
    log_level: int
):
    """Update the basis if (indices, coeffs) represent an independent operator."""
    xmat, xinv = aux
    is_independent, new_xcol = _linear_independence(indices, coeffs, basis_indices, basis_coeffs,
                                                    basis_ptrs, xinv, log_level)
    if not is_independent:
        return basis_indices, basis_coeffs, (xmat, xinv)

    basis_indices, basis_coeffs = _update_basis(indices, coeffs, basis_indices, basis_coeffs,
                                                basis_ptrs, log_level)

    # Add a column to the X matrix
    basis_size = len(basis_ptrs) - 1
    if basis_size > xmat.shape[0]:
        new_xmat = np.eye(xmat.shape[0] + BASIS_ALLOC_UNIT, dtype=xmat.dtype)
        new_xmat[:xmat.shape[0], :xmat.shape[0]] = xmat
        xmat = new_xmat
    xmat[:basis_size - 1, basis_size - 1] = new_xcol
    xmat[basis_size - 1, :basis_size - 1] = new_xcol.conjugate()
    xinv = np.ascontiguousarray(np.linalg.inv(xmat[:basis_size, :basis_size]))

    return basis_indices, basis_coeffs, (xmat, xinv)


@njit
def _if_orthogonal_update(indices, coeffs, basis_indices, basis_coeffs, basis_ptrs, _, log_level):
    o_indices, o_coeffs = _orthogonalize(indices, coeffs, basis_indices, basis_coeffs, basis_ptrs,
                                         log_level)
    if o_indices.shape[0] == 0:
        return basis_indices, basis_coeffs, ()

    o_coeffs /= np.sqrt(np.sum(np.square(np.abs(o_coeffs))))
    basis_indices, basis_coeffs = _update_basis(o_indices, o_coeffs, basis_indices, basis_coeffs,
                                                basis_ptrs, log_level)
    return basis_indices, basis_coeffs, ()


@njit
def _update_loop(
    updater: Callable,
    result_indices: list[np.ndarray],
    result_coeffs: list[np.ndarray],
    basis_indices: np.ndarray,
    basis_coeffs: np.ndarray,
    basis_ptrs: list[int],
    aux: tuple[Any, ...],
    max_dim: int,
    log_level: int
):
    """Loop through the calculated commutators and update the basis with independent elements."""
    # Commutator results are all non-empty and sorted by the indices array so that identical results
    # can be filtered out before invoking the linear dependence check
    prev_indices = np.array([-1], dtype=np.uint64)
    prev_coeffs_conj = None

    num_results = len(result_indices)
    for ires in range(num_results):
        # if log_level <= logging.INFO and ires % 500 == 0:
        #     la_dim = len(basis_ptrs) - 1
        #     with objmode():
        #         LOG.info('Processing commutator %d/%d (Lie algebra dim %d)..',
        #                  ires, num_results, la_dim)

        indices = result_indices[ires]
        coeffs = result_coeffs[ires]
        # Skip if this result is identical to the previous one
        if (prev_indices.shape[0] == indices.shape[0] and np.all(prev_indices == indices)
                and np.isclose(np.abs(prev_coeffs_conj @ coeffs), 1.)):
            continue

        prev_indices = indices
        prev_coeffs_conj = coeffs.conjugate()

        # Check linear independence and update the basis_* arrays
        basis_indices, basis_coeffs, aux = updater(
            indices, coeffs, basis_indices, basis_coeffs, basis_ptrs, aux, log_level
        )

        if len(basis_ptrs) - 1 == max_dim:
            break

    return basis_indices, basis_coeffs, aux


def lie_closure(
    generators: SparsePauliVectorArray,
    *,
    keep_original: bool = True,
    max_dim: Optional[int] = None,
    min_tasks: int = 0,
    max_workers: Optional[int] = None
) -> SparsePauliVectorArray:
    """Compute the Lie closure of given generators.

    Args:
        generators: Lie algebra elements to compute the closure from.
        max_dim: Cutoff for the dimension of the Lie closure. If set, the algorithm may be halted
            before a full closure is obtained.
        min_tasks: Minimum number of commutator calculations to complete before starting a new batch
            of calculations.
        max_workers: Maximun number of threads to use to parallelize the commutator calculations.

    Returns:
        A basis of the Lie closure.
    """
    if len(generators) == 0:
        return generators

    max_dim = max_dim or 4 ** generators.num_qubits - 1

    # Allocate the basis and X arrays and compute the initial basis
    basis = SparsePauliVectorArray([generators[0].normalize()])

    if keep_original:
        # Allocate a large matrix to embed X in
        xmat_size = ((len(basis) - 1) // BASIS_ALLOC_UNIT + 1) * BASIS_ALLOC_UNIT
        xmat = np.eye(xmat_size, dtype=np.complex128)
        xinv = xmat
        updater = _if_independent_update
        aux = (xmat, xinv)
    else:
        updater = _if_orthogonal_update
        aux = ()

    for op in generators[1:]:
        op_coeffs = op.coeffs / np.sqrt(np.sum(np.square(np.abs(op.coeffs))))
        basis.indices, basis.coeffs, aux = updater(
            op.indices, op_coeffs, basis.indices, basis.coeffs, basis.ptrs, aux,
            LOG.getEffectiveLevel()
        )

    if len(basis) >= max_dim:
        return basis

    if max_workers is not None:
        max_workers = min(max_workers, cpu_count())

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = set()

        def calculate_commutators(lhs_first):
            for ib1 in range(lhs_first, len(basis)):
                start, end = basis.ptrs[ib1:ib1 + 2]
                lhs_indices = np.array(basis.indices[start:end])
                lhs_coeffs = np.array(basis.coeffs[start:end])
                for ib2 in range(ib1):
                    start, end = basis.ptrs[ib2:ib2 + 2]
                    rhs_indices = np.array(basis.indices[start:end])
                    rhs_coeffs = np.array(basis.coeffs[start:end])
                    futures.add(
                        executor.submit(
                            _spv_commutator_fast,
                            lhs_indices, lhs_coeffs, rhs_indices, rhs_coeffs, basis.num_qubits, True
                        )
                    )

        # Initial set of commutators of the original generators
        calculate_commutators(1)

        LOG.info('Starting with %d commutators..', len(futures))

        while futures:
            while True:
                done, not_done = wait(futures, return_when=FIRST_COMPLETED)
                if len(not_done) == 0 or len(done) > min_tasks:
                    break
                time.sleep(1.)

            outer_loop_start = time.time()
            LOG.info('Evaluating %d/%d commutators for independence', len(done), len(futures))

            # Pop completed futures out
            futures.difference_update(done)
            # Collect the non-null commutator results
            results = [fut.result() for fut in done if fut.result()[0].shape[0] != 0]
            if len(results) == 0:
                LOG.info('All commutators were null')
                continue
            # Sort the results to make the update loop efficient
            sort_start = time.time()
            indices_tuples = [tuple(res[0]) for res in results]
            sort_idx = sorted(range(len(results)), key=indices_tuples.__getitem__)
            result_indices = [results[idx][0] for idx in sort_idx]
            result_coeffs = [results[idx][1] for idx in sort_idx]
            LOG.debug('Sorted in %.2fs', time.time() - sort_start)

            main_loop_start = time.time()
            old_dim = len(basis)
            if updater is _if_independent_update:
                xinv = np.ascontiguousarray(np.linalg.inv(xmat[:old_dim, :old_dim]))
                aux = (xmat, xinv)

            basis.indices, basis.coeffs, aux = _update_loop(
                updater, result_indices, result_coeffs, basis.indices, basis.coeffs, basis.ptrs,
                aux, max_dim, LOG.getEffectiveLevel()
            )
            new_dim = len(basis)
            LOG.debug('Found %d new ops in %.2fs',
                      new_dim - old_dim, time.time() - main_loop_start)

            if new_dim == max_dim:
                executor.shutdown(wait=False, cancel_futures=True)
                break

            # Calculate the commutators between the new basis elements and all others
            calculate_commutators(old_dim)

            num_added = (new_dim + old_dim - 1) * (new_dim - old_dim) // 2
            LOG.info('Adding %d commutators; %d more to be evaluated', num_added, len(futures))
            if LOG.getEffectiveLevel() < logging.ERROR:
                LOG.info('Current Lie algebra dimension: %d', new_dim)
            LOG.debug('Outer loop took %.2fs', time.time() - outer_loop_start)

    return basis
