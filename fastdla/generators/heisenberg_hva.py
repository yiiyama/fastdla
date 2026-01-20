"""Generators of the Heisenberg model Hamiltonian variational ansatz."""
import numpy as np
from ..sparse_pauli_sum import SparsePauliSum, SparsePauliSumArray


def heisenberg_1d_hva_generators(
    num_spins: int,
    boundary_condition: str = 'periodic'
) -> SparsePauliSumArray:
    r"""Return the generators of 1D Heisenberg model HVA.

    The generators are defined as

    .. math::

        \mathcal{G} = \left\{ \sum_{n} X_n X_{n+1}, \sum_{n} Y_n Y_{n+1},
                              \sum_{n} Z_n Z_{n+1} \right\}.

    The summation is from :math:`n=0` to :math:`N-1` for the periodic boundary condition
    (identifying :math:`X_N, Y_N, Z_N` with :math:`X_0, Y_0, Z_0`), and to :math:`N-2` for the open
    boundary condition.

    Args:
        num_spins: Number of spins in the 1D chain.
        boundary_condition: 'periodic' or 'open'.

    Returns:
        Three generators of the HVA.
    """
    paulis = [[], [], []]

    for iq in range(num_spins - 1):
        paulis[0].append('I' * (num_spins - iq - 2) + 'XX' + 'I' * iq)
        paulis[1].append('I' * (num_spins - iq - 2) + 'YY' + 'I' * iq)
        paulis[2].append('I' * (num_spins - iq - 2) + 'ZZ' + 'I' * iq)

    if boundary_condition == 'periodic':
        paulis[0].append('X' + 'I' * (num_spins - 2) + 'X')
        paulis[1].append('Y' + 'I' * (num_spins - 2) + 'Y')
        paulis[2].append('Z' + 'I' * (num_spins - 2) + 'Z')

    return SparsePauliSumArray([SparsePauliSum(p, np.full(len(p), 1.j)) for p in paulis])


def xxz_hva_generators(
    num_spins: int,
    subspace_controllable: bool = True
) -> SparsePauliSumArray:
    r"""Return the generators of the XXZ model HVA.

    The definition of the generators follows Larocca et al. Quantum 6 (2022):

    .. math::

        \mathcal{G}_{\mathrm{XXZ}_U} & = \left\{
                                        \sum_{n \mathrm{even}} -i (X_n X_{n+1} + Y_n Y_{n+1}),
                                        \sum_{n \mathrm{odd}} -i (X_n X_{n+1} + Y_n Y_{n+1}),
                                        \sum_{n \mathrm{even}} -i Z_n Z_{n+1},
                                        \sum_{n \mathrm{odd}} -i Z_n Z_{n+1}
                                       \right\} \\
        \mathcal{G}_{\mathrm{XXZ}} & = \mathcal{G}_{\mathrm{XXZ}_U} \cup \{-i (Z_0 + Z_{N-1})\}.

    The generators commute with magnetization

    .. math::

        M = \sum_{n=0}^{N-1} Z_n

    and are symmetric under parity

    .. math::

        P: A_n \mapsto A_{N-1-n}.
    """
    paulis = [[], [], [], []]

    for iq in range(0, num_spins, 2):
        paulis[0].append('I' * (num_spins - iq - 2) + 'XX' + 'I' * iq)
        paulis[0].append('I' * (num_spins - iq - 2) + 'YY' + 'I' * iq)
        paulis[2].append('I' * (num_spins - iq - 2) + 'ZZ' + 'I' * iq)

    for iq in range(1, num_spins - 1, 2):
        paulis[1].append('I' * (num_spins - iq - 2) + 'XX' + 'I' * iq)
        paulis[1].append('I' * (num_spins - iq - 2) + 'YY' + 'I' * iq)
        paulis[3].append('I' * (num_spins - iq - 2) + 'ZZ' + 'I' * iq)

    generators = SparsePauliSumArray([SparsePauliSum(p, np.full(len(p), -1.j)) for p in paulis])

    if subspace_controllable:
        paulis = ['I' * (num_spins - 1) + 'Z', 'Z' + 'I' * (num_spins - 1)]
        generators.append(SparsePauliSum(paulis, [-1.j, -1.j]))

    return generators
