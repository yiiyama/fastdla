"""Generators of the transverse-field Ising model HVA."""
import numpy as np
from ..sparse_pauli_vector import SparsePauliVector, SparsePauliVectorArray


def tfim_1d_hva_generators(
    num_spins: int,
    boundary_condition: str = 'open'
) -> SparsePauliVectorArray:
    r"""Return the generators of the transverse-field Ising model HVA.

    The generators are defined as

    .. math::

        \mathcal{G}_{\mathrm{TFIM}} = \left\{
                                        \sum_{n=0}^{N_f} Z_n Z_{n+1},
                                        \sum_{n=0}^{N-1} X_n
                                      \right\}

    with :math:`N_f=N-2 (N-1)` for the open (periodic) boundary conditions (identifying
    :math:`Z_N=Z_0` for the latter).

    Args:
        num_spins: Number of spins in the 1D chain.
        boundary_condition: 'periodic' or 'open'.

    Returns:
        Two generators of the HVA.
    """
    paulis = [[], []]
    for iq in range(num_spins - 1):
        paulis[0].append('I' * (num_spins - iq - 2) + 'ZZ' + 'I' * iq)
        paulis[1].append('I' * (num_spins - iq - 1) + 'X' + 'I' * iq)
    paulis[1].append('X' + 'I' * (num_spins - 1))

    if boundary_condition == 'periodic':
        paulis[0].append('Z' + 'I' * (num_spins - iq - 2) + 'Z')

    return SparsePauliVectorArray([SparsePauliVector(p, np.ones(len(p))) for p in paulis])
