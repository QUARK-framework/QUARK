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

from abc import ABC
import logging
from typing import TypedDict

from quark.utils import start_time_measurement, end_time_measurement
from quark.modules.Core import Core
from quark.modules.applications.qml.Model import Model


class LibraryGenerative(Core, Model, ABC):
    """
    This class is an abstract base class for mapping a library-agnostic gate sequence to a library such as Qiskit.
    It provides no concrete implementations of abstract methods and is intended to be extended by specific libraries.
    """

    def __init__(self, name: str):
        """
        Constructor method.

        :param name: Name of the model
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
