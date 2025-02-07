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

from abc import ABC, abstractmethod


class Model(ABC):
    """
    Abstract base class for any quantum model. This class defines the necessary methods
    that models like 'LibraryGenerative' must implement.
    """

    @abstractmethod
    def sequence_to_circuit(self, input_data: dict) -> dict:
        """
        Abstract method to convert a sequence into a quantum circuit.

        :param input_data: Input data representing the gate sequence
        :return: A dictionary representing the quantum circuit
        """
        pass

    @staticmethod
    @abstractmethod
    def get_execute_circuit(circuit: any, backend: any, config: str, config_dict: dict) -> tuple[any, any]:
        """
        This method combines the circuit implementation and the selected backend and returns a function that will be
        called during training.

        :param circuit: Implementation of the quantum circuit
        :param backend: Configured qiskit backend
        :param config: Name of a backend
        :param config_dict: Dictionary including the number of shots
        :return: Tuple that contains a method that executes the quantum circuit for a given set of parameters and the
        transpiled circuit
        """
        pass

    @staticmethod
    @abstractmethod
    def select_backend(config: str, n_qubits: int) -> any:
        """
        This method configures the backend.

        :param config: Name of a backend
        :param n_qubits: Number of qubits
        :return: Configured backend
        """
        pass
