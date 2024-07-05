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

from typing import TypedDict

from modules.circuits.Circuit import Circuit
from modules.applications.QML.generative_modeling.mappings.LibraryQiskit import LibraryQiskit
from modules.applications.QML.generative_modeling.mappings.PresetQiskitNoisyBackend import PresetQiskitNoisyBackend
from modules.applications.QML.generative_modeling.mappings.CustomQiskitNoisyBackend import CustomQiskitNoisyBackend



class CircuitCardinality(Circuit):
    """
    This class generates a library-agnostic gate sequence, i.e. a list containing information
    about the gates and the wires they act on. 
    The circuit follows the implementation by Gili et al. https://arxiv.org/abs/2207.13645
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__("CircuitCardinality")
        self.submodule_options = ["LibraryQiskit", "CustomQiskitNoisyBackend", "PresetQiskitNoisyBackend"]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this Copula Circuit.

        :return:
                 .. code-block:: python

                     return {
                                "depth": {
                                    "values": [2, 4, 8, 16],
                                    "description": "What depth do you want?"
                                }
                            }

        """
        return {

            "depth": {
                "values": [2, 4, 8, 16],
                "description": "What depth do you want?"
            },
        }

    def get_default_submodule(self, option: str) -> LibraryQiskit:
        if option == "LibraryQiskit":
            return LibraryQiskit()
        elif option == "PresetQiskitNoisyBackend":
            return PresetQiskitNoisyBackend()
        elif option == "CustomQiskitNoisyBackend":
            return CustomQiskitNoisyBackend()
        else:
            raise NotImplementedError(f"Option {option} not implemented")

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python

             depth: int

        """
        depth: int

    def generate_gate_sequence(self, input_data: dict, config: Config) -> dict:
        """
        Returns gate sequence of copula architecture
    
        :param input_data: Collection of information from the previous modules
        :type input_data: dict
        :param config: Config specifying the number of qubits of the circuit
        :type config: Config
        :return: Dictionary including the gate sequence of the Cardinality Circuit
        :rtype: dict
        """
        n_qubits = input_data["n_qubits"]
        depth = config["depth"] // 2

        gate_sequence = []

        for k in range(n_qubits):
            gate_sequence.append(["RX", [k]])
            gate_sequence.append(["RZ", [k]])

        for d in range(depth):
            gate_sequence.append(["Barrier", None])
            for k in range(n_qubits - 1):
                gate_sequence.append(["RXX", [k, k + 1]])
            gate_sequence.append(["Barrier", None])

            if d == depth - 2:
                for k in range(n_qubits):
                    gate_sequence.append(["RX", [k]])
                    gate_sequence.append(["RZ", [k]])
                    gate_sequence.append(["RX", [k]])

            elif d < depth - 2:
                for k in range(n_qubits):
                    gate_sequence.append(["RX", [k]])
                    gate_sequence.append(["RZ", [k]])

        gate_sequence.append(["Barrier", None])

        for k in range(n_qubits):
            gate_sequence.append(["Measure", [k, k]])

        output_dict = {
            "gate_sequence": gate_sequence,
            "circuit_name": "Cardinality",
            "n_qubits": n_qubits,
            "n_registers": None,
            "depth": depth,
            "histogram_train": input_data["histogram_train"],
            "store_dir_iter": input_data["store_dir_iter"],
            "dataset_name": input_data["dataset_name"]
        }

        return output_dict
