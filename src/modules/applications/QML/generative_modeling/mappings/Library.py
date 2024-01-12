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
from qiskit import QuantumCircuit
from qiskit.providers import Backend

from utils import start_time_measurement, end_time_measurement

from modules.Core import Core


class Library(Core, ABC):
    """
    This class is an abstract base class for mapping a library-agnostic gate sequence to a library such as Qiskit 
    """

    def __init__(self, name):
        """
        Constructor method
        """
        self.name = name
        super().__init__()

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python

             backend: str
             n_shots: int

        """
        backend: str
        n_shots: int

    def preprocess(self, input_data: dict, config: Config, **kwargs):
        """
        Base class for mapping the gate sequence to a library such as Qiskit.

        :param input_data: Collection of information from the previous modules
        :type input_data: dict
        :param config: Config specifying the number of qubits of the circuit
        :type config: Config
        :param kwargs: optional keyword arguments
        :type kwargs: dict
        :return: Dictionary including the function to execute the quantum cicrcuit on a simulator or on quantum hardware
        :rtype: (dict, float)
        """

        start = start_time_measurement()

        output = self.sequence_to_circuit(input_data)
        backend = self.select_backend(config["backend"])
        try:
            output["execute_circuit"], output['circuit_transpiled'] = self.get_execute_circuit(
                output["circuit"],
                backend,
                config["backend"],
                config["n_shots"],
                config["transpile_optimization_level"])
        except:
            output["execute_circuit"], output['circuit_transpiled'] = self.get_execute_circuit(
            output["circuit"],
            backend,
            config["backend"],
            config["n_shots"])
        output["backend"] = config["backend"]
        output["n_shots"] = config["n_shots"]
        logging.info("Library created")
        output["store_dir"] = kwargs["store_dir"]

        return output, end_time_measurement(start)

    def postprocess(self, input_data: dict, config: dict, **kwargs):
        """
        This method corresponds to the identity and passes the information of the subsequent module 
        back to the preceding module in the benchmarking process.

        :param input_data: Collected information of the benchmarking process
        :type input_data: dict
        :param config: Config specifying the number of qubits of the circuit
        :type config: Config
        :param kwargs: optional keyword arguments
        :type kwargs: dict
        :return: Same dictionary like input_data with architecture_name
        :rtype: (dict, float)
        """
        start = start_time_measurement()
        return input_data, end_time_measurement(start)

    @abstractmethod
    def sequence_to_circuit(self, input_data):
        pass

    @staticmethod
    @abstractmethod
    def get_execute_circuit(circuit: QuantumCircuit, backend: Backend, config: str, n_shots: int):
        pass

    @staticmethod
    @abstractmethod
    def select_backend(config: str):
        pass
