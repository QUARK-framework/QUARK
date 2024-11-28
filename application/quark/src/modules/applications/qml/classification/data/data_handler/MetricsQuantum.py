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

import math

from typing import Tuple, Dict
import qleet

class MetricsQuantum:
    """
    A class to compute quantum metrics for quantum circuits
    """

    def __init__(
            self,
    ) -> None:
        pass


    def get_metrics(self, circuit, params) -> Dict[str, float]:
        """
        Method that determines all classification metrics

        :param circuit: Quatum circuit
        :return: Dictionary with quantum metrics
        :rtype: dict
        """

        results = {
            "meyer-wallach": self.entanglement_meyer_wallach(circuit, params),
            "expressibility_jsd": self.expressibility_jensen_shannon(circuit, params)
        }

        return results

    def entanglement_meyer_wallach(self, circuit, params, samples=100) -> float:
        """
        Method to determine the Meyer-Wallach entanglement

        :param circuit: Quatum circuit
        :param samples: Samples used to obtain metric
        :return: entanglement
        :rtype: float
        """

        qiskit_descriptor = qleet.interface.circuit.CircuitDescriptor(
            circuit=circuit, params=params, cost_function=None
        )
        qiskit_entanglement_capability = (
            qleet.analyzers.entanglement.EntanglementCapability(
                qiskit_descriptor, samples=samples
            )
        )
        entanglement = qiskit_entanglement_capability.entanglement_capability("meyer-wallach")
        return entanglement
    
    def expressibility_jensen_shannon(self, circuit, params, samples=100) -> float:
        """
        Method to determine the Jensen–Shannon Divergence metric

        :param circuit: Quatum circuit
        :param samples: Samples used to obtain metric
        :return: expr
        :rtype: float
        """

        qiskit_descriptor = qleet.interface.circuit.CircuitDescriptor(
            circuit=circuit, params=params, cost_function=None
        )
        qiskit_expressibility = qleet.analyzers.expressibility.Expressibility(
            qiskit_descriptor, samples=100
        )
        expr = qiskit_expressibility.expressibility("jsd")
        return expr

