"""Implementation of the Lie closure generator using SparsePauliVectors."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Optional
import time
from multiprocessing import cpu_count
import numpy as np
from numba import njit, objmode
from fastdla.sparse_pauli_vector import SparsePauliVector, SparsePauliVectorArray
from fastdla.spv_fast import _uniquify_fast, _spv_commutator_fast, _spv_innerprod_fast

BASIS_ALLOC_UNIT = 1024
MEM_ALLOC_UNIT = SparsePauliVectorArray.MEM_ALLOC_UNIT
DEBUG_NO_JIT = False


def _update_xmatrix(
    xmat: np.ndarray,
    basis_indices: np.ndarray,
    basis_coeffs: np.ndarray,
    basis_ptrs: list[int],
    rows: list[int],
    cols: list[int]
):
    """Update the X matrix of inner products of Lie algebra basis elements.

    The matrix is updated in place.

    Args:
        current: Current matrix.
        basis: The basis (list of linearly independent elements) of the Lie Algebra.
        coordinates: The (row, col) coordinates of the matrix elements to calculate. Only entries
            with col > row are required since the X matrix is Hermitian.
    """
    # Compute the inner products of basis elements for each coordinate.
    for row, col in zip(rows, cols):
        row_start, row_end = basis_ptrs[row:row + 2]
        col_start, col_end = basis_ptrs[col:col + 2]
        ip = _spv_innerprod_fast(
            basis_indices[row_start:row_end],
            basis_coeffs[row_start:row_end],
            basis_indices[col_start:col_end],
            basis_coeffs[col_start:col_end]
        )
        xmat[row, col] = ip
        # Update the transposed coordinate
        xmat[col, row] = ip.conjugate()


if not DEBUG_NO_JIT:
    _update_xmatrix = njit(_update_xmatrix)


def _linear_independence(
    new_indices: np.ndarray,
    new_coeffs: np.ndarray,
    basis_indices: np.ndarray,
    basis_coeffs: np.ndarray,
    basis_ptrs: list[int],
    xinv: np.ndarray
) -> bool:
    """Check linear independence of a new Pauli vector."""
    basis_size = len(basis_ptrs) - 1
    # Create and fill the Π†Q vector
    pidag_q = np.empty(basis_size, dtype=np.complex128)
    is_zero = True
    for ib in range(basis_size):
        start, end = basis_ptrs[ib:ib + 2]
        ip = _spv_innerprod_fast(basis_indices[start:end], basis_coeffs[start:end],
                                 new_indices, new_coeffs)
        pidag_q[ib] = ip
        is_zero &= np.isclose(ip.real, 0.) and np.isclose(ip.imag, 0.)

    if is_zero:
        # Q is orthogonal to all basis vectors
        return True

    # Solve for a: X^{-1}Π†Q
    a_proj = xinv @ pidag_q
    # Residual calculation: uniquify the concatenation of Q and columns of -Pi*ai
    concat_size = new_indices.shape[0]
    nonzero_idx = []
    for ib in range(basis_size):
        a_val = a_proj[ib]
        if not (np.isclose(a_val.real, 0.) and np.isclose(a_val.imag, 0.)):
            nonzero_idx.append(ib)
            concat_size += basis_ptrs[ib + 1] - basis_ptrs[ib]

    concat_indices = np.empty(concat_size, dtype=basis_indices.dtype)
    concat_coeffs = np.empty(concat_size, dtype=basis_coeffs.dtype)
    concat_indices[:new_indices.shape[0]] = new_indices
    concat_coeffs[:new_coeffs.shape[0]] = new_coeffs
    current_pos = new_indices.shape[0]
    for ib in nonzero_idx:
        start, end = basis_ptrs[ib:ib + 2]
        next_pos = current_pos + end - start
        concat_indices[current_pos:next_pos] = basis_indices[start:end]
        concat_coeffs[current_pos:next_pos] = -a_proj[ib] * basis_coeffs[start:end]
        current_pos = next_pos

    indices, _ = _uniquify_fast(concat_indices, concat_coeffs, False)
    return indices.shape[0] != 0


if not DEBUG_NO_JIT:
    _linear_independence = njit(_linear_independence)


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
        xmat = np.eye(len(basis), dtype=basis.coeffs.dtype)
        if len(basis) == 1:
            # _update_xmatrix compilation fails if rows and cols are empty
            xinv = xmat
        else:
            rows, cols = [], []
            for row in range(len(basis) - 1):
                rows += [row] * (len(basis) - 1 - row)
                cols += list(range(row + 1, len(basis)))
            _update_xmatrix(xmat, basis.indices, basis.coeffs, basis.ptrs, rows, cols)
            xinv = np.linalg.inv(xmat)

    return _linear_independence(new_op.indices, new_op.coeffs,
                                basis.indices, basis.coeffs, basis.ptrs,
                                xinv)


def _orthogonalize(
    new_indices: np.ndarray,
    new_coeffs: np.ndarray,
    basis_indices: np.ndarray,
    basis_coeffs: np.ndarray,
    basis_ptrs: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    basis_size = len(basis_ptrs) - 1
    concat_coeffs = np.zeros(basis_coeffs.shape[0] + new_coeffs.shape[0], dtype=basis_coeffs.dtype)

    for ib in range(basis_size):
        start, end = basis_ptrs[ib:ib + 2]
        ip = _spv_innerprod_fast(basis_indices[start:end], basis_coeffs[start:end],
                                 new_indices, new_coeffs)
        if not np.isclose(ip, 0.):
            concat_coeffs[start:end] = -ip * basis_coeffs[start:end]

    concat_coeffs[basis_ptrs[-1]:] = new_coeffs
    concat_indices = np.concatenate((basis_indices, new_indices))
    return _uniquify_fast(concat_indices, concat_coeffs, False)


if not DEBUG_NO_JIT:
    _orthogonalize = njit(_orthogonalize)


def orthogonalize(
    new_op: SparsePauliVector,
    basis: SparsePauliVectorArray
) -> SparsePauliVector:
    """Subtract the subspace projection of an algebra element from itself."""
    indices, coeffs = _orthogonalize(new_op.indices, new_op.coeffs,
                                     basis.indices, basis.coeffs, basis.ptrs)
    return SparsePauliVector(indices, coeffs, num_qubits=new_op.num_qubits, no_check=True)


def _if_independent_update(indices, coeffs, basis_indices, basis_coeffs, basis_ptrs, xmat, xinv):
    """Update the basis if (indices, coeffs) represent an independent operator."""
    if not _linear_independence(indices, coeffs, basis_indices, basis_coeffs, basis_ptrs, xinv):
        return basis_indices, basis_coeffs, xmat, xinv

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

    # Add a column to the X matrix
    basis_size = len(basis_ptrs) - 1
    if basis_size > xmat.shape[0]:
        new_xmat = np.eye(xmat.shape[0] + BASIS_ALLOC_UNIT, dtype=xmat.dtype)
        new_xmat[:xmat.shape[0], :xmat.shape[0]] = xmat
        xmat = new_xmat
    rows = list(range(basis_size - 1))  # rows above the diagonal at column (size - 1)
    cols = [basis_size - 1] * (basis_size - 1)
    _update_xmatrix(xmat, basis_indices, basis_coeffs, basis_ptrs, rows, cols)
    xinv = np.linalg.inv(xmat[:basis_size, :basis_size])
    return basis_indices, basis_coeffs, xmat, xinv


if not DEBUG_NO_JIT:
    _if_independent_update = njit(_if_independent_update)


def _update_loop(
    result_indices: list[np.ndarray],
    result_coeffs: list[np.ndarray],
    basis_indices: np.ndarray,
    basis_coeffs: np.ndarray,
    basis_ptrs: list[int],
    xmat: np.ndarray,
    verbosity: int
):
    """Loop through the calculated commutators and update the basis with independent elements."""
    basis_size = len(basis_ptrs) - 1
    xinv = np.linalg.inv(xmat[:basis_size, :basis_size])
    # Commutator results are all non-empty and sorted by the indices array so that identical results
    # can be filtered out before invoking the linear dependence check
    prev_indices = np.array([-1], dtype=np.uint64)
    prev_coeffs_conj = None

    num_results = len(result_indices)
    for ires in range(num_results):
        if verbosity > 2 and ires % 500 == 0:
            dla_dim = len(basis_ptrs) - 1
            with objmode():
                print(f'Processing commutator {ires}/{num_results} (DLA dim {dla_dim})..',
                      flush=True)

        indices = result_indices[ires]
        coeffs = result_coeffs[ires]
        # Skip if this result is identical to the previous one
        if (prev_indices.shape[0] == indices.shape[0] and np.all(prev_indices == indices)
                and np.isclose(np.abs(prev_coeffs_conj @ coeffs), 1.)):
            continue

        prev_indices = indices
        prev_coeffs_conj = coeffs.conjugate()

        # Check linear independence and update the basis_* arrays
        basis_indices, basis_coeffs, xmat, xinv = _if_independent_update(
            indices, coeffs, basis_indices, basis_coeffs, basis_ptrs, xmat, xinv
        )

    return basis_indices, basis_coeffs, xmat


if not DEBUG_NO_JIT:
    _update_loop = njit(_update_loop)


def lie_closure(
    generators: SparsePauliVectorArray,
    *,
    keep_original: bool = True,
    max_dim: Optional[int] = None,
    verbosity: int = 0,
    min_tasks: int = 0,
    max_workers: Optional[int] = None
) -> SparsePauliVectorArray:
    """Compute the Lie closure of given generators.

    Args:
        generators: Lie algebra elements to compute the closure from.
        max_dim: Cutoff for the dimension of the Lie closure. If set, the algorithm may be halted
            before a full closure is obtained.
        verbosity: Verbosity level.
        min_tasks: Minimum number of commutator calculations to complete before starting a new batch
            of calculations.
        max_workers: Maximun number of threads to use to parallelize the commutator calculations.

    Returns:
        A basis of the Lie closure.
    """
    if len(generators) == 0:
        return generators

    # Allocate the basis and X arrays and compute the initial basis
    basis = SparsePauliVectorArray([generators[0].normalize()])

    if keep_original:
        # Allocate a large matrix to embed X in
        xmat_size = ((len(basis) - 1) // BASIS_ALLOC_UNIT + 1) * BASIS_ALLOC_UNIT
        xmat = np.eye(xmat_size, dtype=np.complex128)
        xinv = xmat
        for op in generators[1:]:
            basis.indices, basis.coeffs, xmat, xinv = _if_independent_update(
                op.indices, op.coeffs, basis.indices, basis.coeffs, basis.ptrs, xmat, xinv
            )
    else:
        raise NotImplementedError()

    max_workers = min(max_workers, cpu_count())
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = set()

        def calculate_commutators(lhs_first):
            for ib1 in range(lhs_first, len(basis)):
                start, end = basis.ptrs[ib1:ib1 + 2]
                lhs_indices = basis.indices[start:end]
                lhs_coeffs = basis.coeffs[start:end]
                for ib2 in range(ib1):
                    start, end = basis.ptrs[ib2:ib2 + 2]
                    rhs_indices = basis.indices[start:end]
                    rhs_coeffs = basis.coeffs[start:end]
                    futures.add(
                        executor.submit(
                            _spv_commutator_fast,
                            lhs_indices, lhs_coeffs, rhs_indices, rhs_coeffs, basis.num_qubits, True
                        )
                    )

        # Initial set of commutators of the original generators
        calculate_commutators(1)

        if verbosity > 1:
            print(f'Starting with {len(futures)} commutators..')

        while futures:
            while True:
                done, not_done = wait(futures, return_when=FIRST_COMPLETED)
                if len(not_done) == 0 or len(done) > min_tasks:
                    break
                time.sleep(1.)

            outer_loop_start = time.time()
            if verbosity > 1:
                print(f'Evaluating {len(done)}/{len(futures)} commutators for independence')

            # Pop completed futures out
            futures.difference_update(done)
            # Collect the non-null commutator results
            results = [fut.result() for fut in done if fut.result()[0].shape[0] != 0]
            if len(results) == 0:
                if verbosity > 1:
                    print('All commutators were null')
                continue
            # Sort the results to make the update loop efficient
            sort_start = time.time()
            indices_tuples = [tuple(res[0]) for res in results]
            sort_idx = sorted(range(len(results)), key=indices_tuples.__getitem__)
            result_indices = [results[idx][0] for idx in sort_idx]
            result_coeffs = [results[idx][1] for idx in sort_idx]
            if verbosity > 2:
                print(f'Sorted in {time.time() - sort_start:.2f}s')

            main_loop_start = time.time()
            old_dim = len(basis)
            basis.indices, basis.coeffs, xmat = _update_loop(
                result_indices, result_coeffs, basis.indices, basis.coeffs, basis.ptrs, xmat,
                verbosity
            )
            new_dim = len(basis)
            if verbosity > 2:
                print(f'Found {new_dim - old_dim} new ops in {time.time() - main_loop_start:.2f}s')

            # Calculate the commutators between the new basis elements and all others
            calculate_commutators(old_dim)

            if verbosity > 1 and new_dim > old_dim:
                num_added = (new_dim + old_dim - 1) * (new_dim - old_dim) // 2
                print(f'Adding {num_added} commutators; total {len(futures)}')
            if verbosity > 0:
                print(f'Current DLA dimension: {new_dim}')
            if verbosity > 2:
                print(f'Outer loop took {time.time() - outer_loop_start:.2f}s')

            if max_dim is not None and new_dim >= max_dim:
                basis.ptrs = basis.ptrs[:max_dim + 1]
                basis.indices = basis.indices[:basis.ptrs[-1]]
                basis.coeffs = basis.coeffs[:basis.ptrs[-1]]
                executor.shutdown(wait=False, cancel_futures=True)
                break

    return basis
