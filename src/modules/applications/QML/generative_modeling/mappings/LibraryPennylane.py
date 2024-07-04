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

from typing import Union

import pennylane as qml
import jax

import numpy as np
from jax import numpy as jnp

jax.config.update("jax_enable_x64", True)

from modules.training.QCBM import QCBM
from modules.training.Inference import Inference
from modules.applications.QML.generative_modeling.mappings.Library import Library


class LibraryPennylane(Library):

    def __init__(self):
        super().__init__("LibraryPennylane")
        self.submodule_options = ["QCBM", "Inference"]

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module

        :return: list of dict with requirements of this module
        :rtype: list[dict]
        """
        return [
            {
                "name": "pennylane",
                "version": "0.29.1"
            },
            {
                "name": "numpy",
                "version": "1.23.5"
            },
            {
                "name": "jax",
                "version": "0.4.5"
            },
        ]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for the PennyLane Library.

        :return:
                 .. code-block:: python

                        return {
                            "backend": {
                                "values": ["default.qubit", "default.qubit.jax", "lightning.qubit", "lightning.gpu"],
                                "description": "Which backend do you want to use?"
                            },

                            "n_shots": {
                                "values": [100, 1000, 10000, 1000000],
                                "description": "How many shots do you want use for estimating the PMF of the model?"
                            }
                        }

        """
        return {
            "backend": {
                "values": ["default.qubit", "default.qubit.jax", "lightning.qubit", "lightning.gpu"],
                "description": "Which device do you want to use?"
            },

            "n_shots": {
                "values": [100, 1000, 10000, 1000000],
                "description": "How many shots do you want use for estimating the PMF of the model?"
            }
        }

    def get_default_submodule(self, training_option: str) -> Union[QCBM, Inference]:

        if training_option == "QCBM":
            return QCBM()
        elif training_option == "Inference":
            return Inference()
        else:
            raise NotImplementedError(f"Training option {training_option} not implemented")
    
    def sequence_to_circuit(self, input_data: dict):
        """
        Method that maps the gate sequence, that specifies the architecture of a quantum circuit
        to its PennyLane implementation.

        :param input_data: Collected information of the benchmarking process
        :type input_data: dict
        :return: Same dictionary but the gate sequence is replaced by its PennyLane implementation
        :rtype: dict
        """
        gate_sequence = input_data["gate_sequence"]
        n_qubits = input_data["n_qubits"]
        num_parameters = sum(1 for gate, _ in gate_sequence if gate in ["RZ", "RX", "RY", "RXX", "RYY", "RZZ", "CRY"])
        def create_circuit(params):
            param_counter = 0
            for gate, wires in gate_sequence:
                if gate == 'Hadamard':
                    qml.Hadamard(wires[0])
                elif gate == 'CNOT':
                    qml.CNOT(wires)
                elif gate == 'RZ':
                    qml.RZ(params[param_counter], wires[0])
                    param_counter += 1
                elif gate == 'RX':
                    qml.RX(params[param_counter], wires[0])
                    param_counter += 1
                elif gate == 'RY':
                    qml.RY(params[param_counter], wires[0])
                    param_counter += 1
                elif gate == "RXX":
                    qml.IsingXX(params[param_counter], wires)
                    param_counter += 1
                elif gate == "RYY":
                    qml.IsingYY(params[param_counter], wires)
                    param_counter += 1
                elif gate == "RZZ":
                    qml.IsingZZ(params[param_counter], wires)
                    param_counter += 1
                elif gate == "CRY":
                    qml.CRY(params[param_counter], wires)
                    param_counter += 1
                elif gate == "Barrier":
                    qml.Barrier()
                elif gate == 'Measure':
                    continue
                else:
                    raise NotImplementedError(f"Gate {gate} not implemented")

            return qml.probs(wires=list(range(n_qubits)))

        input_data["n_params"] = num_parameters
        input_data.pop("gate_sequence")
        input_data["circuit"] = create_circuit

        return input_data

    @staticmethod
    def select_backend(backend_config: str, n_qubits: int) -> qml.device:
        """
        This method configures the backend

        :param config: Name of a backend
        :type config: str
        :return: Configured backend
        :rtype: pennylane.device
        """
        if backend_config == "lightning.gpu":
            backend = qml.device(name="lightning.gpu", wires=n_qubits)

        if backend_config == "lightning.qubit":
            backend = qml.device(name="lightning.qubit", wires=n_qubits)

        elif backend_config == "default.qubit":
            backend = qml.device(name="default.qubit", wires=n_qubits)

        elif backend_config == "default.qubit.jax":
                backend = qml.device(name="default.qubit.jax", wires=n_qubits)

        else:
            raise NotImplementedError(f"Device Configuration {backend_config} not implemented")

        return backend

    @staticmethod
    def get_execute_circuit(circuit: callable, backend: qml.device, backend_config: str, config_dict: dict) -> callable:
        """
        This method combines the PennyLane circuit implementation and the selected backend and returns a function
        that will be called during training.

        :param circuit: PennyLane implementation of the quantum circuit
        :type circuit: callable
        :param backend: Configured PennyLane device
        :type backend: pennylane.device
        :param config: Name of a backend
        :type config: str
        :param n_shots: The number of times the circuit is run
        :type n_shots: int
        :return: Method that executes the quantum circuit for a given set of parameters
        :rtype: callable
        """

        n_shots = config_dict["n_shots"]

        if backend_config == "default.qubit.jax":
            qnode = qml.QNode(circuit, device=backend, interface="jax")
            qnode = jax.jit(qnode)
        else:
            qnode = qml.QNode(circuit, device=backend)

        def execute_circuit(solutions, **kwargs):
            if backend_config == "default.qubit.jax":
                solutions = jnp.asarray(solutions)

            pmfs = []
            samples = []
            for solution in solutions:
                pmf = qnode(solution)
                pmfs.append(pmf)

                if n_shots in list(kwargs.keys()):
                    samples.append(np.random.multinomial(kwargs["n_shots"], pmf))
                else:
                    samples.append(np.random.multinomial(n_shots, pmf))

            return np.asarray(pmfs), np.asarray(samples)

        return execute_circuit, execute_circuit
