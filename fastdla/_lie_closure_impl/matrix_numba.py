"""Implementation of the Lie closure generator using numpy matrices."""
from collections.abc import Sequence
import logging
import time
from typing import Optional
import numpy as np
from numba import njit, objmode
from fastdla.linalg.numba_ops import commutator, innerprod, normalize, orthogonalize
from .algorithms import Algorithms

LOG = logging.getLogger(__name__)
BASIS_ALLOC_UNIT = 1024
_FNS = {}
_INNER_LOOP_FN = {}


def get_algorithm_functions(
    algorithm: Algorithms,
    max_dim: int = 0,
    do_print: bool = False
):
    """Return the component functions for the algorithm."""
    try:
        update_basis, lie_basis, truncate_arrays, resize_arrays = _FNS[algorithm]
    except KeyError:
        @njit(nogil=True)
        def update_basis(
            op: np.ndarray,
            basis_size: int,
            basis: np.ndarray,
            aux: np.ndarray | None
        ) -> tuple[int, np.ndarray, list]:
            norm = np.sqrt(innerprod(op, op).real)
            if np.isclose(norm, 0.):
                return basis_size

            if algorithm == Algorithms.GS_DIRECT:
                orthonormal_basis = basis
            else:
                orthonormal_basis = aux

            op /= norm
            # # Possibly bad shortcut - in other implementations double-orthogonalization is
            # # crucial. Can we really assume single is enough here?
            # orth = orthogonalize(normalize(orthogonalize(op, orthonormal_basis)),
            #                      orthonormal_basis)
            # norm = np.sqrt(innerprod(orth, orth).real)
            # if np.isclose(norm, 1., rtol=1.e-5):
            orth = orthogonalize(op, orthonormal_basis)
            norm = np.sqrt(innerprod(orth, orth).real)
            if not np.isclose(norm, 0., rtol=1.e-5):
                orthonormal_basis[basis_size] = orth / norm
                if algorithm == Algorithms.GRAM_SCHMIDT:
                    basis[basis_size] = op
                basis_size += 1

            return basis_size

        @njit(nogil=True)
        def lie_basis(
            ops: Sequence[np.ndarray]
        ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
            num_ops = ops.shape[0]
            if num_ops == 0:
                raise ValueError('Cannot determine the basis of null space')

            op_shape = ops[0].shape
            max_size = ((num_ops - 1) // BASIS_ALLOC_UNIT + 1) * BASIS_ALLOC_UNIT
            first_op = normalize(ops[0])[None, ...]

            # Set the aux arrays
            if algorithm == Algorithms.GRAM_SCHMIDT:
                # Orthonormal basis
                pad_shape = (max_size - 1,) + op_shape
                aux = np.concatenate((first_op, np.zeros(pad_shape, dtype=ops[0].dtype)), axis=0)
            else:
                aux = np.empty_like(first_op)

            # Initialize a list of normalized generators
            pad_shape = (num_ops - 1,) + op_shape
            basis = np.concatenate((first_op, np.zeros(pad_shape, dtype=ops[0].dtype)), axis=0)
            size = 1
            for op in ops[1:]:
                size = update_basis(normalize(op), size, basis, aux)
            basis = basis[:size]

            return basis, aux

        def truncate_arrays(
            basis: np.ndarray,
            aux: np.ndarray | None,
            size: int
        ):
            """Truncate the basis and auxiliary arrays to the given size."""
            basis = np.asarray(basis[:size])
            if algorithm == Algorithms.GRAM_SCHMIDT:
                aux = np.asarray(aux[:size])

            return basis, aux

        def resize_arrays(
            basis: np.ndarray,
            aux: np.ndarray | None,
            max_size: int
        ):
            """Expand the arrays to a new size."""
            append_shape = (max_size - basis.shape[0],) + basis.shape[1:]
            basis = np.concatenate((basis, np.zeros(append_shape, dtype=basis.dtype)), axis=0)
            if algorithm == Algorithms.GRAM_SCHMIDT:
                aux = np.concatenate((aux, np.zeros(append_shape, dtype=basis.dtype)), axis=0)

            return basis, aux

        _FNS[algorithm] = (update_basis, lie_basis, truncate_arrays, resize_arrays)

    try:
        main_inner_loop = _INNER_LOOP_FN[(algorithm, do_print)]
    except KeyError:
        @njit
        def main_inner_loop(idx_gen, idx_op, basis_size, generators, basis, aux, print_every):
            while idx_op != basis_size and basis_size != max_dim and basis_size != basis.shape[0]:
                if do_print:
                    icomm = idx_op * generators.shape[0] + idx_gen
                    if icomm % print_every == 0:
                        with objmode():
                            LOG.info('Basis size %d; %dth/%d commutator [g[%d], b[%d]]',
                                     basis_size, icomm, basis_size * generators.shape[0],
                                     idx_gen, idx_op)

                basis_size = update_basis(commutator(generators[idx_gen], basis[idx_op]),
                                          basis_size, basis, aux)
                idx_gen = (idx_gen + 1) % generators.shape[0]
                if idx_gen == 0:
                    idx_op += 1

            return idx_gen, idx_op, basis_size

        _INNER_LOOP_FN[(algorithm, do_print)] = main_inner_loop

    if max_dim == 0:
        return lie_basis, truncate_arrays
    return update_basis, lie_basis, truncate_arrays, main_inner_loop, resize_arrays


def lie_basis(
    ops: Sequence[np.ndarray],
    *,
    algorithm: Algorithms = Algorithms.GS_DIRECT,
    return_aux: bool = False
) -> np.ndarray | tuple[np.ndarray, list]:
    """Identify a basis for the linear space spanned by ops.

    Args:
        ops: Lie algebra elements whose span to calculate the basis for.
        algorithm: Algorithm to use for linear independence check and basis update.
        return_aux: Whether to return the auxiliary objects together with the main output.

    Returns:
        A list of linearly independent ops. If return_aux=True, a list of auxiliary objects
        dependent on the algorithm is returned in addition.
    """
    (
        _lie_basis,
        _truncate_arrays
    ) = get_algorithm_functions(algorithm)
    basis, aux = _lie_basis(ops)
    if return_aux:
        basis, aux = _truncate_arrays(basis, aux, basis.shape[0])
        return basis, aux
    return basis


def lie_closure(
    generators: Sequence[np.ndarray],
    *,
    max_dim: Optional[int] = None,
    algorithm: Algorithms = Algorithms.GS_DIRECT,
    return_aux: bool = False,
    print_every: int = 0
) -> np.ndarray | tuple[np.ndarray, list]:
    """Compute the Lie closure of given generators.

    Args:
        generators: Lie algebra elements to compute the closure from.
        max_dim: Cutoff for the dimension of the Lie closure. If set, the algorithm may be halted
            before a full closure is obtained.
        algorithm: Algorithm to use for linear independence check and basis update.
        return_aux: Whether to return the auxiliary objects together with the main output.

    Returns:
        A basis of the Lie algebra spanned by the nested commutators of the generators. If
        return_aux=True, a list of auxiliary objects is returned in addition.
    """
    max_dim = max_dim or generators.shape[-1] ** 2
    log_level = LOG.getEffectiveLevel()
    if print_every == 0:
        if log_level <= logging.DEBUG:
            print_every = 1
        elif log_level <= logging.INFO:  # 20
            print_every = log_level * 100
        else:
            print_every = -1
    do_print = print_every > 0
    (
        _update_basis,
        _lie_basis,
        _truncate_arrays,
        _main_inner_loop,
        _resize_arrays
    ) = get_algorithm_functions(algorithm, max_dim=max_dim, do_print=do_print)

    generators, aux = _lie_basis(generators)
    LOG.info('Number of independent generators: %d', generators.shape[0])

    # Initialize the basis
    basis_size = generators.shape[0]
    max_size = ((basis_size - 1) // BASIS_ALLOC_UNIT + 1) * BASIS_ALLOC_UNIT
    pad_shape = (max_size - generators.shape[0],) + generators.shape[1:]
    basis = np.concatenate([generators, np.zeros(pad_shape, dtype=generators.dtype)], axis=0)

    # First compute the commutators among the generators
    for idx1 in range(generators.shape[0]):
        for idx2 in range(idx1):
            basis_size = _update_basis(commutator(generators[idx1], generators[idx2]),
                                       basis_size, basis, aux)

    # This would be stupid but possible
    if basis_size >= max_dim:
        basis, aux = _truncate_arrays(basis, aux, basis_size)
        if return_aux:
            return basis, aux
        return basis

    # Main loop: generate nested commutators
    idx_gen = 0
    idx_op = generators.shape[0]
    while True:  # Outer loop to handle memory reallocation
        LOG.info('Current Lie algebra dimension: %d', basis_size)
        main_loop_start = time.time()
        idx_gen, idx_op, new_size = _main_inner_loop(idx_gen, idx_op, basis_size,
                                                     generators, basis, aux, print_every)

        LOG.info('Found %d new ops in %.2fs',
                 new_size - basis_size, time.time() - main_loop_start)

        basis_size = new_size
        if idx_op == basis_size or basis_size == max_dim:
            # Computed all commutators
            break

        # Need to resize basis and xmat
        LOG.debug('Resizing basis array to %d', max_size + BASIS_ALLOC_UNIT)

        max_size += BASIS_ALLOC_UNIT
        basis, aux = _resize_arrays(basis, aux, max_size)

    basis, aux = _truncate_arrays(basis, aux, basis_size)
    if return_aux:
        return basis, aux
    return basis
