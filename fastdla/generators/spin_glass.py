"""Generators of the spin glass (SG) model ansatz."""
from typing import Optional
import numpy as np
from ..sparse_pauli_sum import SparsePauliSum, SparsePauliSumArray


def spin_glass_generators(
    num_qubits: int,
    h_mean: float = 0.,
    h_stddev: float = 1.,
    j_mean: float = 0.,
    j_stddev: float = 1.,
    seed: Optional[int] = None
) -> SparsePauliSumArray:
    r"""Construct the generators of the spin glass (SG) model ansatz.

    We adopt the definition of SG in Larocca et al. Quantum 6 (2022):

    .. math::

        & \mathcal{G}_{\mathrm{SG}} = \left\{
                                        \sum_{n=0}^{N_q - 1} i X_n,
                                        \sum_{m<n} i h_m Z_m + i J_{mn} Z_{m} Z_{n}
                                    \right\}, \\
        & h_m, J_{mn} \in \mathbb{R}.

    This set of generators are proven to close to a full rank DLA in the reference.
    """
    rng = np.random.default_rng(seed=seed)

    paulis_zz = []
    paulis_x = []
    for iq in range(num_qubits):
        paulis_x.append('I' * (num_qubits - iq - 1) + 'X' + 'I' * iq)
        for jq in range(iq):
            paulis_zz.append(
                'I' * (num_qubits - iq - 1) + 'Z' + 'I' * (iq - jq - 1) + 'Z' + 'I' * jq
            )
    coeffs_zz = rng.normal(j_mean, j_stddev, len(paulis_zz)) * 1.j
    paulis_z = []
    for iq in range(num_qubits - 1):
        paulis_z.append('I' * (num_qubits - iq - 1) + 'Z' + 'I' * iq)
    coeffs_z = rng.normal(h_mean, h_stddev, num_qubits - 1) * 1.j
    coeffs_z *= np.arange(num_qubits - 1, 0, -1)

    return SparsePauliSumArray([
        SparsePauliSum(paulis_x, np.full(len(paulis_x), 1.j)),
        SparsePauliSum(paulis_z + paulis_zz, np.concatenate([coeffs_z, coeffs_zz]))
    ])
