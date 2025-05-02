#  Copyright 2021 The QUARK Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""This module provides the interface to the circuits that the user specifies and library uses.

This makes the abstractions which ensures all operations in the library are
backend agnostic, by allowing us to get the circuit back in the desired library's
form, qiskit or pytket. It allows the user to specify the circuit in any
form. It also takes the loss function specification as a Pauli String.

It also exposes functions that the user can use to convert their circuits to a
qiskit backend.
WARNING: the conversion is done through a OpenQASM intermediate, operations not
supported on QASM cannot be converted directly, please provide your circuit in a
Qiskit backend in that case.
"""

import typing

import numpy as np
import sympy
import qiskit
import qiskit.quantum_info


def convert_to_qiskit(
    circuit: typing.Union[qiskit.QuantumCircuit]
) -> qiskit.QuantumCircuit:
    """Converts any circuit to qiskit
    :type circuit: Circuit in any supported library
    :param circuit: input circuit in any framework
    :raises ValueError: if the circuit is not from one of the supported frameworks
    :return: circuit in qiskit
    :rtype: qiskit.QuantumCircuit
    """

    if isinstance(circuit, qiskit.QuantumCircuit):
        return circuit
    else:
        raise ValueError(
            f"Expected a circuit object in qiskit, got {type(circuit)}"
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
        circuit: typing.Union[qiskit.QuantumCircuit],
        params: typing.List[typing.Union[sympy.Symbol, qiskit.circuit.Parameter]],
        cost_function: typing.Union[
            qiskit.quantum_info.PauliList, None
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
        if isinstance(self._circuit, qiskit.QuantumCircuit):
            return "qiskit"
        raise ValueError("Unsupported framework of circuit")

    @classmethod
    def from_qasm(
        cls,
        qasm_str: str,
        params: typing.List[typing.Union[sympy.Symbol, qiskit.circuit.Parameter]],
        cost_function: typing.Union[
            qiskit.quantum_info.PauliList, None
        ],
        backend: str = "qiskit",
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
        circuit: typing.Union[qiskit.QuantumCircuit]
        if backend == "qiskit":
            circuit = qiskit.QuantumCircuit.from_qasm_str(qasm_str)
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
    def qiskit_circuit(self) -> qiskit.QuantumCircuit:
        """Get the circuit in qiskit
        :return: the qiskit representation of the circuit
        :rtype: qiskit.QuantumCircuit
        """
        return convert_to_qiskit(self._circuit)

    @property
    def num_qubits(self) -> int:
        """Get the number of qubits for a circuit
        :return: the number of qubits in the circuit
        :rtype: int
        :raises ValueError: if unsupported circuit framework is given
        """
        if isinstance(self._circuit, qiskit.QuantumCircuit):
            return self._circuit.num_qubits
        else:
            raise ValueError("Unsupported framework of circuit")

    def __eq__(self, other: typing.Any) -> bool:
        """Checks equality between a CircuitDescriptor and another object"""
        if isinstance(other, CircuitDescriptor):
            return (
                np.array_equal(self.parameters, other.parameters)
                and self.qiskit_circuit == other.qiskit_circuit
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
