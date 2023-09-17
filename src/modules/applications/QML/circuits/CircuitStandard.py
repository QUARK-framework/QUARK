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

from modules.applications.QML.circuits.Circuit import Circuit
from modules.applications.QML.libraries.LibraryQiskit import LibraryQiskit


class CircuitStandard(Circuit):
    """
    This class generates a library-agnostic gate sequence, i.e. a list containing information
    about the gates and the wires they act on.
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__("DiscreteStandard")
        self.submodule_options = ["LibraryQiskit"]

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module

        :return: list of dict with requirements of this module
        :rtype: list[dict]
        """
        return [
            {
                "name": "scipy",
                "version": "1.11.1"
            }
        ]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this standard circuit.

        :return:
                 .. code-block:: python

                     return {
                                "depth": {
                                    "values": [1, 2, 3, 4, 5],
                                    "description": "What depth do you want?"
                                }
                            }

        """
        return {

            "depth": {
                "values": [1, 2, 3],
                "description": "What depth do you want?"
            }
        }

    def get_default_submodule(self, library_option: str) -> LibraryQiskit:
        if library_option == "LibraryQiskit":
            return LibraryQiskit()
        else:
            raise NotImplementedError(f"Library Option {library_option} not implemented")

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python

             depth: int

        """
        depth: int

    def generate_gate_sequence(self, input_data: dict, config: Config) -> dict:
        """
        Returns gate sequence of standard architecture
    
        :param input_data: Collection of information from the previous modules
        :type input_data: dict
        :param config: Config specifying the number of qubits of the circuit
        :type config: Config
        :return: Dictionary including the gate sequence of the Standard Circuit
        :rtype: dict
        """
        n_registers = input_data["n_registers"]
        n_qubits = input_data["n_qubits"]
        depth = config["depth"]
        n = n_qubits // n_registers

        gate_sequence = []

        for _ in range(depth):
            for k in range(n_qubits):
                gate_sequence.append(["RY", [k]])

            for k in range(n_qubits - 1):
                gate_sequence.append(["RYY", [k, k + 1]])

            for k in range(n_qubits - 1):
                gate_sequence.append(["CRY", [k, k + 1]])

            gate_sequence.append(["Barrier", None])

        for k in range(n_qubits):
            gate_sequence.append(["Measure", [k, k]])

        output_dict = {
            "gate_sequence": gate_sequence,
            "circuit_name": "Standard",
            "n_qubits": n_qubits,
            "n_registers": n_registers,
            "depth": depth,
            "histogram_train": input_data["histogram_train"],
            "store_dir_iter": input_data["store_dir_iter"],
            "dataset_name": input_data["dataset_name"]
        }

        return output_dict
