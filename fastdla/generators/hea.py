"""Generators of the hardware-efficient ansatz (HEA)."""
import numpy as np
from ..sparse_pauli_sum import SparsePauliSum, SparsePauliSumArray


def hea_generators(num_qubits: int) -> SparsePauliSumArray:
    r"""Construct the generators of the hardware-efficient ansatz.

    We adopt the definition of HEA in Larocca et al. Quantum 6 (2022):

    .. math::

        \mathcal{G}_{\mathrm{HEA}} = \left\{ i X_n, i Y_n \right\}_{n=0}^{N_q-1} \cup
                                     \left\{ \sum_{n=0}^{N_q - 2} i Z_n Z_{n+1} \right\}.

    This set of generators are proven to close to a full rank DLA in the reference.
    """
    generators = []

    for iq in range(num_qubits):
        string = 'I' * (num_qubits - iq - 1) + 'X' + 'I' * iq
        generators.append(SparsePauliSum(string, 1.j))
        string = 'I' * (num_qubits - iq - 1) + 'Y' + 'I' * iq
        generators.append(SparsePauliSum(string, 1.j))

    zz_strings = ['I' * (num_qubits - iq - 2) + 'ZZ' + 'I' * iq for iq in range(num_qubits - 1)]
    generators.append(SparsePauliSum(zz_strings, np.full(num_qubits - 1, 1.j)))

    return SparsePauliSumArray(generators)
