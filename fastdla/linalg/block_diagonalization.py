"""Function for simultaneous block diagonalization of matrices."""
import logging
import numpy as np
try:
    import jax.numpy as jnp
    import jax.random as jrandom
except ImportError:
    jnp = None
    jrandom = None

from .gram_schmidt import gram_schmidt, orthonormalize

LOG = logging.getLogger(__name__)


def sbd_fast(
    matrices: np.ndarray,
    hermitian: bool = False,
    krylov_dim: int = 1,
    num_randgen: int = 1,
    seed: int = 0,
    return_blocks: bool = False,
    npmod=np
) -> np.ndarray | tuple[np.ndarray, list[np.ndarray]]:
    r"""Function for simultaneous block diagonalization of matrices.

    This is a numpy port of `sbd_fast.m` from https://github.com/y-z-zhang/net-sync-sym, which is an
    MATLAB implementation of Algorithm 1 in Y. Zhang and A. E. Motter, SIAM Rev. 62, 817-836 (2020).
    A slight modification is added to allow multiple generations of random matrices / vectors to
    avoid being misled by coincidental degeneracies.

    Args:
        matrices: Array of :math:`N \times N` matrices :math:`\{A_k\}_{k=0}^{M-1}` to simultaneously
            block diagonalize. Shape :math:`[M, N, N]`.
        return_blocks: Whether to return the list of resulting blocks.

    Returns:
        The unitary matrix :math:`P` where :math:`P^{\dagger} A_k P` is in a common block diagonal
        form. If `return_blocks` is True, a list of blocks as arrays of shape :math:`[M, N_j, N_j]`
        is additionally returned, with :math:`N_j` the size of the :math:`j` th block.
    """
    LOG.debug('Starting sbd_fast')
    dim = matrices.shape[1]
    if hermitian:
        matrices_combined = matrices
        hpart = matrices
    else:
        hcmatrices = matrices.conjugate().transpose((0, 2, 1))
        matrices_combined = npmod.concatenate([matrices, hcmatrices], axis=0)
        hpart = matrices + hcmatrices
        if matrices.dtype == np.complex128:
            apart = 1.j * (matrices - hcmatrices)

    num_matrices = matrices_combined.shape[0]
    for _ in range(1, krylov_dim):
        raised = npmod.einsum('mij,mjk->mik',
                              matrices_combined[-num_matrices:], matrices_combined[:num_matrices])
        matrices_combined = npmod.concatenate([matrices_combined, raised])

    LOG.debug('%d matrices to test (hermitian=%s)', matrices_combined.shape[0], hermitian)

    if npmod is np:
        rng = np.random.default_rng(seed)

        def normal(shape):
            return rng.normal(size=shape)

    elif npmod is jnp:
        key = jrandom.key(seed)

        def normal(shape):
            nonlocal key
            value = jrandom.normal(key, shape=shape)
            _, key = jrandom.split(key)
            return value

    finest_decomposition = 0
    finest_transform = None

    for irandgen in range(num_randgen):
        LOG.debug('Trial %d', irandgen)
        randmat = npmod.sum(normal((hpart.shape[0], 1, 1)) * hpart, axis=0)
        if matrices.dtype == np.complex128 and not hermitian:
            # pylint: disable-next=used-before-assignment
            randmat += npmod.sum(normal((apart.shape[0], 1, 1)) * apart, axis=0)

        LOG.debug('Diagonalizing the random matrix..')
        _, eigvecs = npmod.linalg.eigh(randmat)
        LOG.debug('Done. %d eigvecs', eigvecs.shape[1])

        transform = npmod.empty((dim, dim), dtype=matrices.dtype)
        num_identified = 0
        evec_idx = 0

        block_sizes = []

        while num_identified < dim:
            LOG.debug('Starting block construction at %dth eigenvector', evec_idx)
            while evec_idx < dim:
                has_orth, vec, _ = orthonormalize(eigvecs[:, evec_idx], transform,
                                                  basis_size=num_identified, npmod=npmod)
                evec_idx += 1
                if has_orth:
                    break
            else:
                raise ValueError('Non-orthogonal eigenvectors? Exhausted eigenvectors of random'
                                 ' matrix before completing the unitary.')

            LOG.debug('Found a new dimension at %dth eigenvector. Performing Gram-Schmidt..',
                      evec_idx)

            block_basis = npmod.empty_like(transform)
            if npmod is np:
                block_basis[0] = vec
            elif npmod is jnp:
                block_basis = block_basis.at[0].set(vec)
            block_basis, basis_size = gram_schmidt(matrices_combined @ vec, basis=block_basis,
                                                   basis_size=1, npmod=npmod)

            LOG.debug('Done. New block basis size %d', basis_size)

            while True:
                LOG.debug('Performing Gram-Schmidt on random vectors from the block basis..')
                vrand = npmod.sum(block_basis[:basis_size] * normal((basis_size, 1)), axis=0)
                if matrices.dtype == np.complex128:
                    vrand += 1.j * npmod.sum(block_basis[:basis_size] * normal((basis_size, 1)),
                                             axis=0)

                current_size = basis_size
                block_basis, basis_size = gram_schmidt(matrices_combined @ vrand, basis=block_basis,
                                                       basis_size=current_size, npmod=npmod)
                LOG.debug('Done. New block basis size %d', basis_size)
                if basis_size == current_size:
                    break

            current_num = num_identified
            num_identified += basis_size
            if npmod is np:
                transform[current_num:num_identified] = block_basis[:basis_size]
            elif npmod is jnp:
                transform = transform.at[current_num:num_identified].set(block_basis[:basis_size])

            block_sizes.append(basis_size)

        if len(block_sizes) > finest_decomposition:
            LOG.debug('Found the finest block decomposition so far: block sizes %s', block_sizes)
            finest_decomposition = len(block_sizes)
            if return_blocks:
                blocks = []
                permutation = []
                for block_idx in np.argsort(block_sizes):
                    start = sum(block_sizes[:block_idx])
                    indices = list(range(start, start + block_sizes[block_idx]))
                    block_basis = transform[indices]
                    blocks.append(
                        npmod.einsum('ij,mjk,lk->mil',
                                     block_basis.conjugate(), matrices, block_basis)
                    )
                    permutation.extend(indices)
                finest_transform = transform[permutation]
            else:
                finest_transform = transform

    if return_blocks:
        return finest_transform.T, blocks
    return finest_transform.T
