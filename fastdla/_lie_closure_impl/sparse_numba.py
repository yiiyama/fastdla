"""Implementation of the Lie closure generator using SparsePauliVectors."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import logging
from typing import Optional
import time
from multiprocessing import cpu_count
import numpy as np
from numba import njit, objmode
from fastdla.sparse_pauli_vector import SparsePauliVector, SparsePauliVectorArray
from fastdla.spv_fast import (abs_square, complex_isclose, _uniquify_fast, _spv_commutator_fast,
                              _spv_dot_fast)

LOG = logging.getLogger(__name__)
BASIS_ALLOC_UNIT = 1024
MEM_ALLOC_UNIT = SparsePauliVectorArray.MEM_ALLOC_UNIT


@njit
def _orthogonalize(
    new_indices: np.ndarray,
    new_coeffs: np.ndarray,
    basis_indices: np.ndarray,
    basis_coeffs: np.ndarray,
    basis_ptrs: list[int],
    normalize: bool,
    log_level: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the component of a vector with respect to an orthonormal basis."""
    # Strategy: Calculate v - sum_j (v.b_j)b_j through uniquify()
    basis_size = len(basis_ptrs) - 1
    concat_size = new_indices.shape[0]
    nonzero_idx = []
    ips = []
    for ib in range(basis_size):
        start, end = basis_ptrs[ib:ib + 2]
        ip = _spv_dot_fast(basis_indices[start:end], basis_coeffs[start:end],
                           new_indices, new_coeffs)
        if ip.real != 0. or ip.imag != 0.:
            nonzero_idx.append(ib)
            ips.append(ip)
            concat_size += basis_ptrs[ib + 1] - basis_ptrs[ib]

    LOG.debug(ips)
    # if log_level <= logging.DEBUG:
    #     support_dim = len(nonzero_idx)
    #     with objmode():
    #         LOG.debug('Vector has %d-dim support in the basis', support_dim)

    concat_indices = np.empty(concat_size, dtype=basis_indices.dtype)
    concat_coeffs = np.empty(concat_size, dtype=basis_coeffs.dtype)
    concat_indices[:new_indices.shape[0]] = new_indices
    concat_coeffs[:new_coeffs.shape[0]] = new_coeffs
    current_pos = new_indices.shape[0]
    for ib, ip in zip(nonzero_idx, ips):
        start, end = basis_ptrs[ib:ib + 2]
        next_pos = current_pos + end - start
        concat_indices[current_pos:next_pos] = basis_indices[start:end]
        concat_coeffs[current_pos:next_pos] = -ip * basis_coeffs[start:end]
        current_pos = next_pos

    return _uniquify_fast(concat_indices, concat_coeffs, False)


def orthogonalize(
    new_op: SparsePauliVector,
    basis: SparsePauliVectorArray,
    normalize: bool = True
) -> SparsePauliVector:
    """Subtract the subspace projection of an algebra element from itself."""
    indices, coeffs = _orthogonalize(new_op.indices, new_op.coeffs, basis.indices, basis.coeffs,
                                     basis.ptrs, normalize, LOG.getEffectiveLevel())
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
        if log_level <= logging.DEBUG:
            with objmode():
                LOG.debug('Expanded the basis array by %d', additional_capacity)

    basis_ptrs.append(next_ptr)
    basis_indices[basis_ptrs[-2]:basis_ptrs[-1]] = indices
    basis_coeffs[basis_ptrs[-2]:basis_ptrs[-1]] = coeffs

    return basis_indices, basis_coeffs


@njit
def _if_independent_update(
    indices: np.ndarray,
    coeffs: np.ndarray,
    isource: tuple[int, int],
    basis_indices: np.ndarray,
    basis_coeffs: np.ndarray,
    basis_ptrs: list[int],
    log_level: int
) -> tuple[bool, np.ndarray, np.ndarray]:
    o_indices, o_coeffs = _orthogonalize(indices, coeffs, basis_indices, basis_coeffs, basis_ptrs,
                                         False, log_level)
    if o_indices.shape[0] == 0:
        # if log_level <= logging.DEBUG:
        #     basis_size = len(basis_ptrs) - 1
        #     with objmode():
        #         LOG.debug('New op has no orthogonal component; not updating basis (size %d)',
        #                   basis_size)

        return False, basis_indices, basis_coeffs

    o_norm = np.sqrt(np.sum(abs_square(o_coeffs)))
    # if not np.isclose(o_norm, 1., rtol=1.8):
    #     # Distill the orthogonal component
    #     o_indices, o_coeffs = _orthogonalize(o_indices, o_coeffs / o_norm, basis_indices,
    #                                          basis_coeffs, basis_ptrs, False, log_level)

    if log_level <= logging.DEBUG:
        basis_size = len(basis_ptrs) - 1
        with objmode():
            LOG.debug('Commutator [%d, %d] has an orthogonal component with norm %.3e;'
                      ' updating basis (size %d)', isource[0], isource[1], o_norm, basis_size)

    o_coeffs /= o_norm
    basis_indices, basis_coeffs = _update_basis(o_indices, o_coeffs, basis_indices, basis_coeffs,
                                                basis_ptrs, log_level)
    return True, basis_indices, basis_coeffs


@njit
def _update_loop(
    result_indices: list[np.ndarray],
    result_coeffs: list[np.ndarray],
    result_isource: list[tuple[int, int]],
    basis_indices: np.ndarray,
    basis_coeffs: np.ndarray,
    basis_ptrs: list[int],
    max_dim: int,
    log_level: int
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Loop through the calculated commutators and update the basis with independent elements."""
    independent_elements = []
    # Commutator results are all non-empty and sorted by the indices array so that identical results
    # can be filtered out before invoking the linear dependence check
    prev_indices = np.array([-1], dtype=np.uint64)
    prev_coeffs_conj = None

    num_results = len(result_indices)
    for ires in range(num_results):
        if log_level <= logging.INFO and ires % 500 == 0:
            la_dim = len(basis_ptrs) - 1
            ilhs, irhs = result_isource[ires]
            with objmode():
                LOG.info('Processing commutator [%d, %d] (%d/%d). Lie algebra dim %d',
                         ilhs, irhs, ires, num_results, la_dim)

        indices = result_indices[ires]
        coeffs = result_coeffs[ires]
        # Skip if this result is identical to the previous one
        if (prev_indices.shape[0] == indices.shape[0] and np.all(prev_indices == indices)
                and np.isclose(np.abs(prev_coeffs_conj @ coeffs), 1.)):
            continue

        prev_indices = indices
        prev_coeffs_conj = coeffs.conjugate()
        isource = result_isource[ires]

        # Check linear independence and update the basis_* arrays
        updated, basis_indices, basis_coeffs = _if_independent_update(
            indices, coeffs, isource, basis_indices, basis_coeffs, basis_ptrs, log_level
        )
        if updated:
            independent_elements.append(ires)

        if len(basis_ptrs) - 1 == max_dim:
            break

    return basis_indices, basis_coeffs, independent_elements


@njit(nogil=True)
def _commutator(ilhs, lhs_indices, lhs_coeffs, irhs, rhs_indices, rhs_coeffs, num_qubits):
    """Calculate the commutator and return the results together with the row and column numbers."""
    indices, coeffs = _spv_commutator_fast(lhs_indices, lhs_coeffs, rhs_indices, rhs_coeffs,
                                           num_qubits, True)
    return indices, coeffs, ilhs, irhs


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
        A list of linearly independent nested commutators and the orthonormal basis if
        keep_original=True, otherwise only the orthonormal basis.
    """
    if len(generators) == 0:
        return generators

    max_dim = max_dim or 4 ** generators.num_qubits - 1

    # Allocate the basis and X arrays and compute the initial basis
    basis = SparsePauliVectorArray([generators[0].normalize()])

    if keep_original:
        nested_commutators = SparsePauliVectorArray([basis[0]])
        source = nested_commutators
    else:
        source = basis

    for iop in range(1, len(generators)):
        op = generators[iop]
        op = op.normalize()
        updated, basis.indices, basis.coeffs = _if_independent_update(
            op.indices, op.coeffs, (iop, -1), basis.indices, basis.coeffs, basis.ptrs,
            LOG.getEffectiveLevel()
        )
        if updated and keep_original:
            nested_commutators.append(op)

    if len(basis) >= max_dim:
        return basis

    if max_workers is not None:
        max_workers = min(max_workers, cpu_count())

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = set()

        def calculate_commutators(lhs_first):
            for ib1 in range(lhs_first, len(source)):
                start, end = source.ptrs[ib1:ib1 + 2]
                lhs_indices = np.array(source.indices[start:end])
                lhs_coeffs = np.array(source.coeffs[start:end])
                for ib2 in range(ib1):
                    start, end = source.ptrs[ib2:ib2 + 2]
                    rhs_indices = np.array(source.indices[start:end])
                    rhs_coeffs = np.array(source.coeffs[start:end])
                    futures.add(
                        executor.submit(
                            _commutator,
                            ib1, lhs_indices, lhs_coeffs,
                            ib2, rhs_indices, rhs_coeffs,
                            basis.num_qubits
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
            result_isource = [results[idx][2:] for idx in sort_idx]
            LOG.debug('Sorted in %.2fs', time.time() - sort_start)

            main_loop_start = time.time()
            old_dim = len(basis)
            basis.indices, basis.coeffs, independent_elements = _update_loop(
                result_indices, result_coeffs, result_isource, basis.indices, basis.coeffs,
                basis.ptrs, max_dim, LOG.getEffectiveLevel()
            )
            new_dim = len(basis)
            if keep_original:
                for ires in independent_elements:
                    nested_commutators.append(
                        SparsePauliVector(result_indices[ires], result_coeffs[ires],
                                          num_qubits=basis.num_qubits)
                    )
            if LOG.getEffectiveLevel() <= logging.DEBUG:
                LOG.debug('Found %d new ops in %.2fs',
                          new_dim - old_dim, time.time() - main_loop_start)
                comms = [f'[{result_isource[ires][0]}, {result_isource[ires][1]}]'
                         for ires in independent_elements]
                LOG.debug(', '.join(f'{comm}->{new}'
                                    for comm, new in zip(comms, range(old_dim, new_dim))))

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

    if keep_original:
        return nested_commutators, basis
    return basis
