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

from abc import ABC, abstractmethod
import logging
from typing import TypedDict

from utils import start_time_measurement, end_time_measurement
from modules.Core import Core


class Library(Core, ABC):
    """
    This class is an abstract base class for mapping a library-agnostic gate sequence to a library such as Qiskit.
    """

    def __init__(self, name: str):
        """
        Constructor method.
        """
        self.name = name
        super().__init__()

    class Config(TypedDict):
        """
        Attributes of a valid config.

        .. code-block:: python

             backend: str
             n_shots: int

        """
        backend: str
        n_shots: int

    def preprocess(self, input_data: dict, config: Config, **kwargs) -> tuple[dict, float]:
        """
        Base class for mapping the gate sequence to a library such as Qiskit.

        :param input_data: Collection of information from the previous modules
        :param config: Config specifying the number of qubits of the circuit
        :param kwargs: Optional keyword arguments
        :return: Tuple including dictionary with the function to execute the quantum circuit on a simulator or quantum
                 hardware and the computation time of the function
        """
        start = start_time_measurement()

        output = self.sequence_to_circuit(input_data)
        backend = self.select_backend(config["backend"], output["n_qubits"])
        output["execute_circuit"], output['circuit_transpiled'] = self.get_execute_circuit(
            output["circuit"],
            backend,
            config["backend"],
            config
        )
        output["backend"] = config["backend"]
        output["n_shots"] = config["n_shots"]
        logging.info("Library created")
        output["store_dir"] = kwargs["store_dir"]

        return output, end_time_measurement(start)

    def postprocess(self, input_data: dict, config: dict, **kwargs) -> tuple[dict, float]:
        """
        This method corresponds to the identity and passes the information of the subsequent module 
        back to the preceding module in the benchmarking process.

        :param input_data: Collected information of the benchmarking procesS
        :param config: Config specifying the number of qubits of the circuit
        :param kwargs: Optional keyword arguments
        :return: Tuple with input dictionary and the computation time of the function
        """
        start = start_time_measurement()
        return input_data, end_time_measurement(start)

    @abstractmethod
    def sequence_to_circuit(self, input_data: dict) -> dict:
        pass

    @staticmethod
    @abstractmethod
    def get_execute_circuit(circuit: any, backend: any, config: str, config_dict: dict) -> tuple[any, any]:
        """
        This method combines the circuit implementation and the selected backend and returns a function that will be
        called during training.

        :param circuit: Implementation of the quantum circuiT
        :param backend: Configured backend
        :param config: Name of the PennyLane devicE
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
        return
