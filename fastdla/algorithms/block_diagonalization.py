"""Function for simultaneous block diagonalization of matrices."""
import logging
import numpy as np
try:
    import jax.numpy as jnp
    import jax.random as jrandom
except ImportError:
    jnp = None
    jrandom = None
from fastdla.algorithms.gram_schmidt import gram_schmidt

LOG = logging.getLogger(__name__)


def sbd_fast(
    matrices: np.ndarray,
    hermitian: bool = False,
    krylov_dim: int = 1,
    num_randgen: int = 1,
    seed: int = 0,
    orth_cutoff: float = 1.e-08,
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
        seeds = npmod.linalg.eigh(randmat)[1].T
        LOG.debug('Done. %d eigvecs', seeds.shape[0])

        transform = npmod.empty((dim, dim), dtype=matrices.dtype)
        basis_size = 0
        block_sizes = []
        while basis_size < dim:
            LOG.debug('Starting with transform matrix size: %d', basis_size)
            seed = seeds[basis_size]

            if npmod is np:
                transform[basis_size] = seed
            elif npmod is jnp:
                transform = transform.at[basis_size].set(seed)
            transform, new_size = gram_schmidt(matrices_combined @ seed, basis=transform,
                                               basis_size=basis_size + 1, cutoff=orth_cutoff,
                                               npmod=npmod)

            block_size = new_size - basis_size
            LOG.debug('Done checking matrices @ seed. New block basis size %d', block_size)

            num_no_change = 0
            while num_no_change < 3:
                LOG.debug('Performing Gram-Schmidt on random vectors from the block basis..')
                vrand = npmod.sum(transform[basis_size:new_size] * normal((block_size, 1)),
                                  axis=0)
                if matrices.dtype == np.complex128:
                    vrand += npmod.sum(transform[basis_size:new_size] * normal((block_size, 1)),
                                       axis=0) * 1.j

                current_size = new_size
                transform, new_size = gram_schmidt(matrices_combined @ vrand, basis=transform,
                                                   basis_size=current_size, cutoff=orth_cutoff,
                                                   npmod=npmod)
                block_size = new_size - basis_size
                LOG.debug('Done checking random (iteration %d). New block basis size %d',
                          num_no_change, block_size)
                if new_size == current_size:
                    num_no_change += 1
                else:
                    num_no_change = 0
            LOG.debug('Block basis size plateaued. Saving the block and re-orthogonalizing seeds.')

            block_sizes.append(block_size)
            basis_size = new_size

            # Orthogonalize the seeds wrt current basis
            # If randmat is non-degenerate, the identified basis vectors so far in transform should
            # be orthogonal to the remaining seeds. (We have identified a subspace that is spanned
            # by some combination of the eigenvectors of the input matrices, which would therefore
            # be orthogonal to eigenvectors that do not belong to the subspace). When there are
            # degeneracies, the basis for the degenerate subspace that eigh chooses may not coincide
            # with the subspace structure. We can use another round of Gram-Schmidt to separate the
            # seed eigenvectors in and out of the identified subspaces.
            # The degeneracy of randmat can occur if the matrices have shared degeneracies or under
            # some special configurations.
            # Also, by performing GS here, the next seed is trivially given by seeds[num_identified]
            # Argument `basis`` must be given as a copy of transform because the returned array is
            # the same object as this input in the numpy implementation.
            seeds = gram_schmidt(seeds, basis=transform.copy(), basis_size=basis_size,
                                 cutoff=orth_cutoff)[0]

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
