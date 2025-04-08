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

"""Module to evaluate the achievable entanglement in circuits."""

import itertools
import typing

from qiskit_aer.noise import NoiseModel as qiskitNoiseModel

from qiskit.quantum_info import partial_trace
from scipy.special import comb

import numpy as np

from ..interface.metas import MetaExplorer
from ..interface.circuit import CircuitDescriptor
from ..simulators.circuit_simulators import CircuitSimulator

NOISE_MODELS = {
    "qiskit": qiskitNoiseModel,
}


class EntanglementCapability(MetaExplorer):
    """Calculates entangling capability of a parameterized quantum circuit"""

    def __init__(
        self,
        circuit: CircuitDescriptor,
        noise_model: typing.Union[
            qiskitNoiseModel, None
        ] = None,
        samples: int = 1000,
    ):
        """Constructor for entanglement capability plotter

        :param circuit: input circuit as a CircuitDescriptor object
        :param noise_model:  (dict, NoiseModel) initialization noise-model dictionary for
            generating noise model
        :param samples: number of samples for the experiment
        :returns Entanglement object instance
        :raises ValueError: If circuit and noise model does not correspond to same framework
        """
        super().__init__()
        self.circuit = circuit

        if noise_model is not None:
            if (
                (
                    circuit.default_backend == "qiskit"
                    and isinstance(noise_model, qiskitNoiseModel)
                )  
            ):
                self.noise_model = noise_model
            else:
                raise ValueError(
                    f"Circuit and noise model must correspond to the same \
                    framework but circuit:{circuit.default_backend} and \
                    noise_model:{type(noise_model)} were provided."
                )
        else:
            self.noise_model = None

        self.num_samples = samples

    def gen_params(self) -> typing.Tuple[typing.List, typing.List]:
        """Generate parameters for the calculation of expressibility

        :return theta (np.array): first list of parameters for the parameterized quantum circuit
        :return phi (np.array): second list of parameters for the parameterized quantum circuit
        """
        theta = [
            {p: 2 * np.random.random() * np.pi for p in self.circuit.parameters}
            for _ in range(self.num_samples)
        ]
        phi = [
            {p: 2 * np.random.random() * np.pi for p in self.circuit.parameters}
            for _ in range(self.num_samples)
        ]
        return theta, phi

    @staticmethod
    def scott_helper(state, perms):
        """Helper function for entanglement measure. It gives trace of the output state"""
        dems = np.linalg.matrix_power(
            [partial_trace(state, list(qb)).data for qb in perms], 2
        )
        trace = np.trace(dems, axis1=1, axis2=2)
        return np.sum(trace).real

    def meyer_wallach_measure(self, states, num_qubits):
        r"""Returns the meyer-wallach entanglement measure for the given circuit.

        .. math::
            Q = \frac{2}{|\vec{\theta}|}\sum_{\theta_{i}\in \vec{\theta}}
            \Bigg(1-\frac{1}{n}\sum_{k=1}^{n}Tr(\rho_{k}^{2}(\theta_{i}))\Bigg)

        """
        permutations = list(itertools.combinations(range(num_qubits), num_qubits - 1))
        ns = 2 * sum(
            [
                1 - 1 / num_qubits * self.scott_helper(state, permutations)
                for state in states
            ]
        )
        return ns.real

    def scott_measure(self, states, num_qubits):
        r"""Returns the scott entanglement measure for the given circuit.

        .. math::
            Q_{m} = \frac{2^{m}}{(2^{m}-1) |\vec{\theta}|}\sum_{\theta_i \in \vec{\theta}}\
            \bigg(1 - \frac{m! (n-m)!)}{n!}\sum_{|S|=m} \text{Tr} (\rho_{S}^2 (\theta_i)) \bigg)\
            \quad m= 1, \ldots, \lfloor n/2 \rfloor

        """
        m = range(1, num_qubits // 2 + 1)
        permutations = [
            list(itertools.combinations(range(num_qubits), num_qubits - idx))
            for idx in m
        ]
        combinations = [1 / comb(num_qubits, idx) for idx in m]
        contributions = [2**idx / (2**idx - 1) for idx in m]
        ns = []

        for ind, perm in enumerate(permutations):
            ns.append(
                contributions[ind]
                * sum(
                    [
                        1 - combinations[ind] * self.scott_helper(state, perm)
                        for state in states
                    ]
                )
            )

        return np.array(ns)

    def entanglement_capability(
        self, measure: str = "meyer-wallach", shots: int = 1024
    ) -> float:
        """Returns entanglement measure for the given circuit

        :param measure: specification for the measure used in the entangling capability
        :param shots: number of shots for circuit execution
        :returns pqc_entangling_capability (float): entanglement measure value
        :raises ValueError: if invalid measure is specified
        """
        thetas, phis = self.gen_params()

        theta_circuits = [
            CircuitSimulator(self.circuit, self.noise_model).simulate(theta, shots)
            for theta in thetas
        ]
        phi_circuits = [
            CircuitSimulator(self.circuit, self.noise_model).simulate(phi, shots)
            for phi in phis
        ]

        num_qubits = self.circuit.num_qubits

        if measure == "meyer-wallach":
            pqc_entanglement_capability = self.meyer_wallach_measure(
                theta_circuits + phi_circuits, num_qubits
            ) / (2 * self.num_samples)
        elif measure == "scott":
            pqc_entanglement_capability = self.scott_measure(
                theta_circuits + phi_circuits, num_qubits
            ) / (2 * self.num_samples)
        else:
            raise ValueError(
                "Invalid measure provided, choose from 'meyer-wallach' or 'scott'"
            )

        return pqc_entanglement_capability
