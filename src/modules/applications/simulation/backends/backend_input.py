from dataclasses import dataclass
from qiskit import QuantumCircuit


@dataclass
class BackendInput:
    """
    Input required for a quantum backend.
    """
    circuits: list[QuantumCircuit]
