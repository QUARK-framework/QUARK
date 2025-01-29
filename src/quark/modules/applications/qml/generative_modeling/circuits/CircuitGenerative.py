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
from quark.modules.Core import Core
from quark.utils import start_time_measurement, end_time_measurement

from quark.modules.applications.qml.Circuit import Circuit


class CircuitGenerative(Circuit, Core, ABC):
    """
    This module is abstract base class for the library-agnostic gate sequence, that define a quantum circuit.
    """

    def __init__(self, name: str):
        """
        Constructor method.

        :param name: The name of the circuit architecture
        """
        super().__init__()
        self.architecture_name = name

    def preprocess(self, input_data: dict, config: dict, **kwargs) -> tuple[dict, float]:
        """
        Library-agnostic implementation of the gate sequence, that will be mapped to backend such as Qiskit in the
         subsequent module.

        :param input_data: Collection of information from the previous modules
        :param config: Config specifying the number of qubits of the circuit
        :param kwargs: Optional keyword arguments
        :return: Dictionary including the dataset, the gate sequence needed for circuit construction, and the time it
                 took generate the gate sequence.
        """
        start = start_time_measurement()
        circuit_constr = self.generate_gate_sequence(input_data, config)

        if "generalization_metrics" in input_data:
            circuit_constr["generalization_metrics"] = input_data["generalization_metrics"]

        return circuit_constr, end_time_measurement(start)

    def postprocess(self, input_data: dict, config: dict, **kwargs) -> tuple[dict, float]:
        """
        Method that passes back information of the subsequent modules to the preceding modules.

        :param input_data: Collected information of the benchmarking process
        :param config: Config specifying the number of qubits of the circuit
        :param kwargs: Optional keyword arguments
        :return: Same dictionary like input_data with architecture_name and execution time
        """
        start = start_time_measurement()
        input_data["architecture_name"] = self.architecture_name
        return input_data, end_time_measurement(start)
