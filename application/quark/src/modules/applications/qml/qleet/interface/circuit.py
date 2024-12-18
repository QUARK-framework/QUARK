"""This module provides the interface to the circuits that the user specifies and library uses.

This makes the abstractions which ensures all operations in the library are
backend agnostic, by allowing us to get the circuit back in the desired library's
form, cirq, qiskit or pytket. It allows the user to specify the circuit in any
form. It also takes the loss function specification as a Pauli String.

It also exposes functions that the user can use to convert their circuits to a
qiskit or cirq backend.
WARNING: the conversion is done through a OpenQASM intermediate, operations not
supported on QASM cannot be converted directly, please provide your circuit in a
Cirq or Qiskit backend in that case.
"""

import typing

import numpy as np
import sympy

import cirq
import qiskit
import pyquil

from cirq.contrib.qasm_import import circuit_from_qasm
from cirq.contrib.quil_import import circuit_from_quil
import qiskit.quantum_info
import pyquil.paulis


def convert_to_cirq(
    circuit: typing.Union[qiskit.QuantumCircuit, cirq.Circuit, pyquil.Program]
) -> cirq.Circuit:
    """Converts any circuit to cirq
    :type circuit: Circuit in any supported library
    :param circuit: input circuit in any framework
    :return: circuit in cirq
    :rtype: cirq.Circuit
    :raises ValueError: if the circuit is not from one of the supported frameworks
    """
    if isinstance(circuit, cirq.Circuit):
        return circuit
    elif isinstance(circuit, qiskit.QuantumCircuit):
        return circuit_from_qasm(circuit.qasm())
    elif isinstance(circuit, pyquil.Program):
        return circuit_from_quil(str(circuit))
    else:
        raise ValueError(
            f"Expected a circuit object in cirq, qiskit or pyquil, got {type(circuit)}"
        )


def convert_to_qiskit(
    circuit: typing.Union[qiskit.QuantumCircuit, cirq.Circuit, pyquil.Program]
) -> qiskit.QuantumCircuit:
    """Converts any circuit to qiskit
    :type circuit: Circuit in any supported library
    :param circuit: input circuit in any framework
    :raises ValueError: if the circuit is not from one of the supported frameworks
    :return: circuit in qiskit
    :rtype: qiskit.QuantumCircuit
    """
    if isinstance(circuit, cirq.Circuit):
        return qiskit.QuantumCircuit.from_qasm_str(circuit.to_qasm())
    elif isinstance(circuit, qiskit.QuantumCircuit):
        return circuit
    elif isinstance(circuit, pyquil.Program):
        return convert_to_qiskit(convert_to_cirq(circuit))
    else:
        raise ValueError(
            f"Expected a circuit object in cirq, qiskit or pyquil, got {type(circuit)}"
        )


def convert_to_pyquil(
    circuit: typing.Union[qiskit.QuantumCircuit, cirq.Circuit, pyquil.Program]
) -> qiskit.QuantumCircuit:
    """Converts any circuit to pyquil
    :type circuit: Circuit in any supported library
    :param circuit: input circuit in any framework
    :raises ValueError: if the circuit is not from one of the supported frameworks
    :return: circuit in pyquil
    :rtype: pyquil.Program
    """
    if isinstance(circuit, cirq.Circuit):
        return pyquil.Program(circuit.to_quil())
    elif isinstance(circuit, qiskit.QuantumCircuit):
        return pyquil.Program(convert_to_cirq(circuit).to_quil())
    elif isinstance(circuit, pyquil.Program):
        return circuit
    else:
        raise ValueError(
            f"Expected a circuit object in cirq, qiskit or pyquil, got {type(circuit)}"
        )


class CircuitDescriptor:
    """The interface for users to provide a circuit in any framework and visualize it in qLEET.

    It consists of 3 parts:
    * Circuit: which has the full ansatz preparation from the start where
    * Params: list of parameters which are used to parameterize the circuit
    * Cost Function: presently a pauli string, which we measure to get the
        output we are optimizing over

    Combined they form the full the parameterized quantum circuit from the initial qubits to the end
    measurement.
    """

    def __init__(
        self,
        circuit: typing.Union[qiskit.QuantumCircuit, cirq.Circuit, pyquil.Program],
        params: typing.List[typing.Union[sympy.Symbol, qiskit.circuit.Parameter]],
        cost_function: typing.Union[
            cirq.PauliSum, qiskit.quantum_info.PauliList, pyquil.paulis.PauliSum, None
        ] = None,
    ):
        """Constructor for the CircuitDescriptor

        :type circuit: Circuit in any supported library
        :param circuit: The full circuit which generates the required quantum state
        :type params: list[sympy.Symbol]
        :param params: The list of parameters to optimize over
        :type cost_function: PauliSum in any supported library
        :param cost_function: The measurement operation as a PauliString

        If you are not providing the full list of parameters of the circuit because
        you don't want to optimize over some of those parameters, because use a
        Parameter Resolver to resolve those parameter values before you pass in the
        lists. The list of parameters passed in here ought to be complete.
        """
        self._circuit = circuit
        self._params = params
        self._cost = cost_function

    @property
    def default_backend(self) -> str:
        """Returns the backend in which the user had provided the circuit.
        :returns: The name of the default backend
        :rtype: str
        :raises ValueError: if the given circuit is not from a supported library
        """
        if isinstance(self._circuit, cirq.Circuit):
            return "cirq"
        if isinstance(self._circuit, qiskit.QuantumCircuit):
            return "qiskit"
        if isinstance(self._circuit, pyquil.Program):
            return "pyquil"
        raise ValueError("Unsupported framework of circuit")

    @classmethod
    def from_qasm(
        cls,
        qasm_str: str,
        params: typing.List[typing.Union[sympy.Symbol, qiskit.circuit.Parameter]],
        cost_function: typing.Union[
            cirq.PauliSum, qiskit.quantum_info.PauliList, pyquil.paulis.PauliSum, None
        ],
        backend: str = "cirq",
    ):
        """Generate the descriptor from OpenQASM string

        :type qasm_str: str
        :param qasm_str: OpenQASM string for each part of the circuit
        :type params: list[sympy.Symbol]
        :param params: list of sympy symbols which act as parameters for the PQC
        :type cost_function: PauliSum
        :param cost_function: pauli-string operator to implement cost function
        :type backend: str
        :param backend: backend for the circuit descriptor objects
        :return: The CircuitDescriptor object
        :rtype: CircuitDescriptor
        :raises ValueError: if one of the 3 supported backends is not the input
        """
        circuit: typing.Union[cirq.Circuit, qiskit.QuantumCircuit, pyquil.Program]
        if backend == "cirq":
            circuit = circuit_from_qasm(qasm_str)
        elif backend == "qiskit":
            circuit = qiskit.QuantumCircuit.from_qasm_str(qasm_str)
        elif backend == "pyquil":
            circuit = pyquil.Program(circuit_from_qasm(qasm_str).to_quil())
        else:
            raise ValueError()

        return CircuitDescriptor(
            circuit=circuit, params=params, cost_function=cost_function
        )

    @property
    def parameters(
        self,
    ) -> typing.List[typing.Union[sympy.Symbol, qiskit.circuit.Parameter]]:
        """The list of sympy symbols to resolve as parameters, will be swept from 0 to 2*pi
        :return: list of parameters
        """
        return self._params

    def __len__(self) -> int:
        """Number of parameters in the variational circuit
        :return: number of parameters in the circuit
        """
        return len(self.parameters)

    @property
    def cirq_circuit(self) -> cirq.Circuit:
        """Get the circuit in cirq
        :return: the cirq representation of the circuit
        :rtype: cirq.Circuit
        """
        return convert_to_cirq(self._circuit)

    @property
    def qiskit_circuit(self) -> qiskit.QuantumCircuit:
        """Get the circuit in qiskit
        :return: the qiskit representation of the circuit
        :rtype: qiskit.QuantumCircuit
        """
        return convert_to_qiskit(self._circuit)

    @property
    def pyquil_circuit(self) -> pyquil.Program:
        """Get the circuit in pyquil
        :return: the pyquil representation of the circuit
        :rtype: pyquil.Program
        """
        return convert_to_pyquil(self._circuit)

    @property
    def num_qubits(self) -> int:
        """Get the number of qubits for a circuit
        :return: the number of qubits in the circuit
        :rtype: int
        :raises ValueError: if unsupported circuit framework is given
        """
        if isinstance(self._circuit, cirq.Circuit):
            return len(self._circuit.all_qubits())
        elif isinstance(self._circuit, qiskit.QuantumCircuit):
            return self._circuit.num_qubits
        elif isinstance(self._circuit, pyquil.Program):
            return len(self._circuit.get_qubits())
        else:
            raise ValueError("Unsupported framework of circuit")

    @property
    def cirq_cost(self) -> cirq.PauliSum:
        """Returns the cost function, which is a function that takes in the state vector or the
        density matrix and returns the loss value of the solution envisioned by the Quantum Circuit.
        :raises ValueError: if the circuit is not from one of the supported frameworks
        :raises NotImplementedError: Long as qiskit and pyquil ports of pauli-string aren't written
        :return: cost function
        TODO: Implement conversions into Cirq PauliSum
        """
        if isinstance(self._cost, cirq.PauliSum):
            return self._cost
        elif isinstance(self._cost, qiskit.quantum_info.PauliList):
            raise NotImplementedError("Qiskit PauliString support is not implemented")
        elif isinstance(self._cost, pyquil.paulis.PauliSum):
            raise NotImplementedError("PyQuil PauliString support is not implemented")
        else:
            raise ValueError("Cost object should be a Pauli-Sum object")

    def __eq__(self, other: typing.Any) -> bool:
        """Checks equality between a CircuitDescriptor and another object"""
        if isinstance(other, CircuitDescriptor):
            return (
                np.array_equal(self.parameters, other.parameters)
                and self.cirq_circuit == other.cirq_circuit
            )
        return False

    def __repr__(self) -> str:
        """Prints the representation of the CircuitDescriptor
        You can eval this to get the object back.

        :returns: The repr string
        :rtype: str
        """
        return f"qleet.CircuitDescriptor({repr(self._circuit)}, {repr(self._params)})"

    def __str__(self) -> str:
        """Prints the string form of the CircuitDescriptor

        :returns: The string form
        :rtype: str
        """
        return f"qleet.CircuitDescriptor({repr(self._circuit)})"
