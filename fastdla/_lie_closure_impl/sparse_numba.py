"""Implementation of the Lie closure generator using SparsePauliSums."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import logging
import copy
from typing import Optional
import time
from multiprocessing import cpu_count
import numpy as np
from numba import njit, objmode
from fastdla.sparse_pauli_sum import SparsePauliSum, SparsePauliSumArray
from fastdla.sps_fast import (abs_square, sps_commutator_fast, _uniquify_fast, _sps_commutator_fast,
                              _sps_dot_fast, _spsarray_append_fast)
from fastdla._lie_closure_impl.algorithms import Algorithms

LOG = logging.getLogger(__name__)
BASIS_ALLOC_UNIT = 1024
MEM_ALLOC_UNIT = SparsePauliSumArray.MEM_ALLOC_UNIT


@njit
def _orthogonalize(
    new_indices: np.ndarray,
    new_coeffs: np.ndarray,
    basis_indices: np.ndarray,
    basis_coeffs: np.ndarray,
    basis_ptrs: list[int],
    normalize: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the component of a vector with respect to an orthonormal basis."""
    # Strategy: Calculate v - sum_j (v.b_j)b_j through uniquify()
    basis_size = len(basis_ptrs) - 1
    concat_size = new_indices.shape[0]
    nonzero_idx = []
    ips = []
    for ib in range(basis_size):
        start, end = basis_ptrs[ib:ib + 2]
        ip = _sps_dot_fast(basis_indices[start:end], basis_coeffs[start:end],
                           new_indices, new_coeffs)
        # Checking exact equality with zero because _sps_dot_fast rounds off close-to-zero ips
        if not (ip.real == 0. and ip.imag == 0.):
            nonzero_idx.append(ib)
            ips.append(ip)
            concat_size += basis_ptrs[ib + 1] - basis_ptrs[ib]

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

    return _uniquify_fast(concat_indices, concat_coeffs, normalize)


@njit
def _if_independent_update(
    indices: np.ndarray,
    coeffs: np.ndarray,
    basis_indices: np.ndarray,
    basis_coeffs: np.ndarray,
    basis_ptrs: list[int],
    log_level: int
) -> tuple[bool, np.ndarray, np.ndarray]:
    o_indices, o_coeffs = _orthogonalize(indices, coeffs, basis_indices, basis_coeffs, basis_ptrs,
                                         False)
    if o_indices.shape[0] == 0:
        return False, basis_indices, basis_coeffs

    o_norm = np.sqrt(np.sum(abs_square(o_coeffs)))
    o_coeffs /= o_norm
    # A single pass can result in a false orthogonal vector due to finite numerical precision
    # If the extracted vector is truly orthogonal, renormalizing it and further extracting the
    # orthogonal component should result in almost-unit-norm vector.
    if not np.isclose(o_norm, 1., rtol=1.e-5):
        # Distill the orthogonal component
        o_indices, o_coeffs = _orthogonalize(o_indices, o_coeffs, basis_indices, basis_coeffs,
                                             basis_ptrs, False)
        o_norm = np.sqrt(np.sum(abs_square(o_coeffs)))
        if not np.isclose(o_norm, 1., rtol=1.e-5):
            # We had a false orthogonal vector
            return False, basis_indices, basis_coeffs

        o_coeffs /= o_norm

    if log_level <= logging.DEBUG:
        basis_size = len(basis_ptrs)  # need to do this outside of objmode()
        with objmode():
            LOG.debug('Updating basis size to %d', basis_size)

    basis_indices, basis_coeffs = _spsarray_append_fast(basis_indices, basis_coeffs, basis_ptrs,
                                                        o_indices, o_coeffs)
    return True, basis_indices, basis_coeffs


def if_independent_update(
    op: SparsePauliSum,
    basis: SparsePauliSumArray,
    aux: list[SparsePauliSumArray] | None,
    log_level: int
) -> bool:
    updated, basis.indices, basis.coeffs = _if_independent_update(
        op.indices, op.coeffs, basis.indices, basis.coeffs, basis.ptrs, log_level
    )
    if updated and aux:
        aux[0].append(op)
    return updated


@njit
def _update_loop(
    result_indices: list[np.ndarray],
    result_coeffs: list[np.ndarray],
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
        if log_level <= logging.INFO and ires % 2000 == 0:
            la_dim = len(basis_ptrs) - 1
            with objmode():
                LOG.info('Processing commutator %d/%d. Lie algebra dim %d',
                         ires, num_results, la_dim)

        indices = result_indices[ires]
        coeffs = result_coeffs[ires]
        # Skip if this result is identical to the previous one
        if (prev_indices.shape[0] == indices.shape[0] and np.all(prev_indices == indices)
                and np.isclose(np.abs(prev_coeffs_conj @ coeffs), 1.)):
            continue

        prev_indices = indices
        prev_coeffs_conj = coeffs.conjugate()

        # Check linear independence and update the basis_* arrays
        updated, basis_indices, basis_coeffs = _if_independent_update(
            indices, coeffs, basis_indices, basis_coeffs, basis_ptrs, log_level
        )
        if updated:
            independent_elements.append(ires)
        if len(basis_ptrs) - 1 == max_dim:
            break

    return basis_indices, basis_coeffs, independent_elements


def _lie_basis(
    ops: SparsePauliSumArray,
    *,
    algorithm: Algorithms = Algorithms.GS_DIRECT
):
    if len(ops) == 0:
        raise ValueError('Cannot determine the basis of null space')

    # Allocate the basis and X arrays and compute the initial basis
    basis = SparsePauliSumArray([ops[0].normalize()])
    aux = []
    if algorithm == Algorithms.GRAM_SCHMIDT:
        aux.append(SparsePauliSumArray([basis[0]]))

    for iop in range(1, len(ops)):
        op = ops[iop].normalize()
        if_independent_update(op, basis, aux, LOG.getEffectiveLevel())

    return basis, aux


def _truncate_arrays(
    basis: np.ndarray,
    aux: list,
    size: int,
    *,
    algorithm: Algorithms = Algorithms.GS_DIRECT
):
    basis = basis[:size]
    if algorithm == Algorithms.GRAM_SCHMIDT:
        aux[0] = aux[0][:size]
    return basis, aux


def lie_basis(
    ops: SparsePauliSumArray,
    *,
    algorithm: Algorithms = Algorithms.GS_DIRECT,
    return_aux: bool = False
) -> SparsePauliSumArray | tuple[SparsePauliSumArray, list]:
    """Compute a basis of the linear space spanned by ops.

    Args:
        ops: Lie algebra elements whose span to calculate the basis for.
        algorithm: Algorithm to use for linear independence check and basis update.
        return_aux: Whether to return the auxiliary objects together with the main output.

    Returns:
        A list of linearly independent ops. If return_aux=True, a list of auxiliary objects
        dependent on the algorithm is returned in addition.
    """
    basis, aux = _lie_basis(ops, algorithm=algorithm)
    if return_aux:
        return basis, aux
    return basis


def lie_closure(
    generators: SparsePauliSumArray,
    *,
    max_dim: Optional[int] = None,
    algorithm: Algorithms = Algorithms.GS_DIRECT,
    return_aux: bool = False,
    min_tasks: int = 0,
    max_workers: Optional[int] = None
) -> tuple[SparsePauliSumArray, SparsePauliSumArray] | SparsePauliSumArray:
    """Compute the Lie closure of given generators.

    Args:
        generators: Lie algebra elements to compute the closure from.
        max_dim: Cutoff for the dimension of the Lie closure. If set, the algorithm may be halted
            before a full closure is obtained.
        algorithm: Algorithm to use for linear independence check and basis update.
        return_aux: Whether to return the auxiliary objects together with the main output.
        min_tasks: Minimum number of commutator calculations to complete before starting a new batch
            of calculations.
        max_workers: Maximun number of threads to use to parallelize the commutator calculations.

    Returns:
        A basis of the Lie algebra spanned by the nested commutators of the generators. If
        return_aux=True, a list of auxiliary objects is returned in addition.
    """
    if algorithm not in (Algorithms.GS_DIRECT, Algorithms.GRAM_SCHMIDT):
        raise NotImplementedError(f'Algorithm {algorithm} is not implemented in sparse_numba')

    generators, aux = _lie_basis(generators, algorithm=algorithm)
    num_gen = len(generators)
    LOG.info('Number of independent generators: %d', num_gen)

    basis = copy.deepcopy(generators)

    # Compute the commutators among elements of source[1]
    for iop1 in range(num_gen):
        for iop2 in range(iop1):
            comm = sps_commutator_fast(generators[iop1], generators[iop2], True)
            if_independent_update(comm, basis, aux, LOG.getEffectiveLevel())

    max_dim = max_dim or 4 ** generators.num_qubits - 1

    if len(basis) >= max_dim:
        basis, aux = _truncate_arrays(basis, aux, max_dim)
        if return_aux:
            return basis, aux
        return basis

    if max_workers is not None:
        max_workers = min(max_workers, cpu_count())

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = set()

        def calculate_commutators(start):
            for iopl in range(start, len(basis)):
                lhs = basis[iopl]
                for iopr in range(num_gen):
                    rhs = generators[iopr]
                    futures.add(
                        executor.submit(_sps_commutator_fast,
                                        lhs.indices, lhs.coeffs, rhs.indices, rhs.coeffs,
                                        basis.num_qubits, True)
                    )

        # Initial set of commutators of the original generators
        calculate_commutators(num_gen)

        LOG.info('Starting with %d commutators..', len(futures))

        while futures:
            while True:
                done, not_done = wait(futures, return_when=FIRST_COMPLETED)
                if len(not_done) == 0 or len(done) > min_tasks:
                    break
                time.sleep(1.)

            LOG.info('Current Lie algebra dimension: %d', len(basis))
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
            LOG.debug('Sorted %d results in %.2fs', len(results), time.time() - sort_start)

            main_loop_start = time.time()
            old_dim = len(basis)
            if algorithm == Algorithms.GS_DIRECT:
                out = basis
            else:
                out = aux[0]
            out.indices, out.coeffs, independent_elements = _update_loop(
                result_indices, result_coeffs, out.indices, out.coeffs,
                basis.ptrs, max_dim, LOG.getEffectiveLevel()
            )
            new_dim = len(out)
            if algorithm == Algorithms.GRAM_SCHMIDT:
                for ires in independent_elements:
                    basis.append(result_indices[ires], result_coeffs[ires])

            if LOG.getEffectiveLevel() <= logging.DEBUG:
                LOG.debug('Found %d new ops in %.2fs',
                          new_dim - old_dim, time.time() - main_loop_start)

            if new_dim == max_dim:
                executor.shutdown(wait=False, cancel_futures=True)
                break

            if new_dim > old_dim:
                # Calculate the commutators between the new basis elements and all others
                calculate_commutators(old_dim)

                num_added = (new_dim + old_dim - 1) * (new_dim - old_dim) // 2
                LOG.info('Adding %d commutators; %d more to be evaluated', num_added, len(futures))

    if return_aux:
        return basis, aux
    return basis
