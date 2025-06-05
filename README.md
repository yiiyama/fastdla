# fastdla

Fast Lie closure (aka dynamical Lie algebra) calculation using JAX / Numba with algorithms described
in [arXiv:2506.01120](https://arxiv.org/abs/2506.01120).

## Installation

```pip install fastdla```

## Example usage

The following lines of code calculates the DLA for a 10-qubit Hardware efficient ansatz defined in [Larocca et al. Quantum 6 (2022)](https://quantum-journal.org/papers/q-2022-09-29-824/):

```
from fastdla import lie_closure
from fastdla.generators.hea import hea_generators

generators = hea_generators(num_qubits=10)
dla = lie_closure(generators)
```

## Documentation

[Read the Docs](https://fastdla.readthedocs.io/en/latest/)
