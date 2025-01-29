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

from typing import Union, TypedDict

from quark.modules.applications.qml.generative_modeling.circuits.CircuitGenerative import CircuitGenerative
from quark.modules.applications.qml.generative_modeling.mappings.LibraryQiskit import LibraryQiskit
from quark.modules.applications.qml.generative_modeling.mappings.LibraryPennylane import LibraryPennylane
from quark.modules.applications.qml.generative_modeling.mappings.PresetQiskitNoisyBackend import PresetQiskitNoisyBackend
from quark.modules.applications.qml.generative_modeling.mappings.CustomQiskitNoisyBackend import CustomQiskitNoisyBackend


class CircuitStandard(CircuitGenerative):
    """
    This class generates a library-agnostic gate sequence, i.e. a list containing information
    about the gates and the wires they act on.
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__("DiscreteStandard")
        self.submodule_options = [
            "LibraryQiskit",
            "LibraryPennylane",
            "CustomQiskitNoisyBackend",
            "PresetQiskitNoisyBackend"
        ]

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module.

        :return: List of dict with requirements of this module
        """
        return []

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this standard circuit.

        :return: Dictionary of parameter options.
        .. code-block:: python

            return {
                    "depth": {
                        "values": [1, 2, 3],
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

    def get_default_submodule(self, option: str) \
            -> Union[LibraryQiskit, LibraryPennylane, PresetQiskitNoisyBackend, CustomQiskitNoisyBackend]:
        """
        Returns the default submodule based on the given option.

        :param option: The submodule option to select
        :return: Instance of the selected submodule
        :raises NotImplemented: If the provided option is not implemented
        """
        if option == "LibraryQiskit":
            return LibraryQiskit()
        if option == "LibraryPennylane":
            return LibraryPennylane()
        elif option == "PresetQiskitNoisyBackend":
            return PresetQiskitNoisyBackend()
        elif option == "CustomQiskitNoisyBackend":
            return CustomQiskitNoisyBackend()
        else:
            raise NotImplementedError(f"Option {option} not implemented")

    class Config(TypedDict):
        """
        Attributes of a valid config.

        .. code-block:: python

             depth: int

        """
        depth: int

    def generate_gate_sequence(self, input_data: dict, config: Config) -> dict:
        """
        Returns gate sequence of standard architecture.

        :param input_data: Collection of information from the previous modules
        :param config: Config specifying the number of qubits of the circuit
        :return: Dictionary including the gate sequence of the Standard Circuit
        """
        n_registers = input_data["n_registers"]
        n_qubits = input_data["n_qubits"]
        depth = config["depth"]

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
            "train_size": input_data["train_size"],
            "dataset_name": input_data["dataset_name"],
            "binary_train": input_data["binary_train"]
        }

        return output_dict
