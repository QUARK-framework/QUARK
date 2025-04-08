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

"""Module to evaluate the expressibility of circuits."""

import itertools
# import typing
from typing import Union, List, Tuple, cast

from qiskit_aer.noise import NoiseModel as qiskitNoiseModel
#from cirq.devices.noise_model import NoiseModel as cirqNoiseModel
from pyquil.noise import NoiseModel as pyquilNoiseModel

from qiskit.quantum_info import state_fidelity
from scipy.spatial.distance import jensenshannon

import matplotlib.pyplot as plt
import numpy as np

from ..interface.metas import MetaExplorer
from ..interface.circuit import CircuitDescriptor
from ..simulators.circuit_simulators import CircuitSimulator

NOISE_MODELS = {
    #"cirq": cirqNoiseModel,
    "pyquil": pyquilNoiseModel,
    "qiskit": qiskitNoiseModel,
}


class Expressibility(MetaExplorer):
    """Calculates expressibility of a parameterized quantum circuit"""

    def __init__(
        self,
        circuit: CircuitDescriptor,
        noise_model: Union[
            #cirqNoiseModel, 
            qiskitNoiseModel, pyquilNoiseModel, None
        ] = None,
        samples: int = 1000,
    ):
        """Constructor the the Expressibility analyzer

        :param circuit: input circuit as a CircuitDescriptor object
        :param noise_model:  (dict, NoiseModel) initialization noise-model dictionary
        :param samples: number of samples for the experiment
        :raises ValueError: If circuit and noise model does not correspond to same framework
        """
        super().__init__()
        self.circuit = circuit

        if noise_model is not None:
            if (
                (
                    #circuit.default_backend == "cirq"
                    #and isinstance(noise_model, cirqNoiseModel)
                #)
                #or (
                    circuit.default_backend == "qiskit"
                    and isinstance(noise_model, qiskitNoiseModel)
                )
                or (
                    circuit.default_backend == "pyquil"
                    and isinstance(noise_model, pyquilNoiseModel)
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
        self.expr = 0.0
        self.plot_data: List[np.ndarray] = []

    @staticmethod
    def kl_divergence(prob_a: np.ndarray, prob_b: np.ndarray) -> float:
        """Returns KL divergence between two probabilities"""
        prob_a[prob_a == 0] = 1e-10
        kl_div = np.sum(np.where(prob_a != 0, prob_a * np.log(prob_a / prob_b), 0))
        return cast(float, kl_div)

    def gen_params(self) -> Tuple[List, List]:
        """Generate parameters for the calculation of expressibility

        :returns theta (np.array): first list of parameters for the parameterized quantum circuit
        :returns phi (np.array): second list of parameters for the parameterized quantum circuit
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

    def prob_haar(self) -> np.ndarray:
        """Returns probability density function of fidelities for Haar Random States"""
        fidelity = np.linspace(0, 1, self.num_samples)
        num_qubits = self.circuit.num_qubits
        return (2**num_qubits - 1) * (1 - fidelity + 1e-8) ** (2**num_qubits - 2)

    def prob_pqc(self, shots: int = 1024) -> np.ndarray:
        """Return probability density function of fidelities for PQC

        :param shots: number of shots for circuit execution
        :returns fidelities (np.array): np.array of fidelities
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
        fidelity = np.array(
            [
                state_fidelity(rho_a, rho_b)
                for rho_a, rho_b in itertools.product(theta_circuits, phi_circuits)
            ]
        )
        return np.array(fidelity)

    def expressibility(self, measure: str = "kld", shots: int = 1024) -> float:
        r"""Returns expressibility for the circuit

        .. math::
            Expr = D_{KL}(\hat{P}_{PQC}(F; \theta) | P_{Haar}(F))\\
            Expr = D_{\sqrt{JSD}}(\hat{P}_{PQC}(F; \theta) | P_{Haar}(F))

        :param measure: specification for the measure used in the expressibility calculation
        :param shots: number of shots for circuit execution
        :returns pqc_expressibility: float, expressibility value
        :raises ValueError: if invalid measure is specified
        """
        haar = self.prob_haar()
        haar_prob: np.ndarray = haar / float(haar.sum())

        if len(self.circuit.parameters) > 0:
            fidelity = self.prob_pqc(shots)
        else:
            fidelity = np.ones(self.num_samples**2)

        bin_edges: np.ndarray
        pqc_hist, bin_edges = np.histogram(
            fidelity, self.num_samples, range=(0, 1), density=True
        )
        pqc_prob: np.ndarray = pqc_hist / float(pqc_hist.sum())

        if measure == "kld":
            pqc_expressibility = self.kl_divergence(pqc_prob, haar_prob)
        elif measure == "jsd":
            pqc_expressibility = 1 - jensenshannon(pqc_prob, haar_prob, 2.0)
        else:
            raise ValueError("Invalid measure provided, choose from 'kld' or 'jsd'")
        self.plot_data = [haar_prob, pqc_prob, bin_edges]
        self.expr = pqc_expressibility

        return pqc_expressibility

    def compare_expressibility(self, circuit: Union[CircuitDescriptor, List[CircuitDescriptor]], measure: str = "kld", shots: int = 1024) -> List[float]:
        r"""Compares expressibility against the provided circuit

        .. math::
            Expr = D_{KL}(\hat{P}_{PQC_1}(F; \theta) | \hat{P}_{PQC_2}(F; \theta))\\
            Expr = D_{\sqrt{JSD}}(\hat{P}_{PQC_1}(F; \theta) | \hat{P}_{PQC_2}(F; \theta))

        :param measure: specification for the measure used in the expressibility calculation
        :param shots: number of shots for circuit execution
        :returns pqc_expressibility: float, expressibility value
        :raises ValueError: if invalid measure is specified
        """

        thetas, phis = self.gen_params()
        fidelities = []
        pqc_probs = []

        if not isinstance(circuit, list) and isinstance(circuit, CircuitDescriptor):
            circuit = [circuit]

        for circ in [*circuit, self.circuit]:

            if len(circuit.parameters) > 0:
                theta_circuits = [
                    CircuitSimulator(circ, self.noise_model).simulate(theta, shots)
                    for theta in thetas
                ]
                phi_circuits = [
                    CircuitSimulator(circ, self.noise_model).simulate(phi, shots)
                    for phi in phis
                ]
                fidelity = np.array(
                    [
                        state_fidelity(rho_a, rho_b)
                        for rho_a, rho_b in itertools.product(theta_circuits, phi_circuits)
                    ]
                )
            else:
                fidelity = np.ones(self.num_samples**2)
            
            fidelities.append(fidelity)
            bin_edges: np.ndarray
            pqc_hist, bin_edges = np.histogram(
                fidelity, self.num_samples, range=(0, 1), density=True
            )
            pqc_prob: np.ndarray = pqc_hist / float(pqc_hist.sum())
            pqc_probs.append(pqc_probs)

        pqc_expressibilities = []
        for pqc_prob in pqc_probs[:-1]:
            if measure == "kld":
                pqc_expressibility = self.kl_divergence(pqc_prob, pqc_probs[-1])
            elif measure == "jsd":
                pqc_expressibility = jensenshannon(pqc_prob, pqc_probs[-1], 2.0)
            else:
                raise ValueError("Invalid measure provided, choose from 'kld' or 'jsd'")
            pqc_expressibilities.appned(pqc_expressibility)

        return pqc_expressibilities

    def plot(self, figsize=(6, 4), dpi=300, **kwargs):
        """Returns plot for expressibility visualization"""
        if not self.plot_data:
            raise ValueError("Perform expressibility calculation first")

        haar_prob, pqc_prob, bin_edges = self.plot_data
        expr = self.expr

        bin_middles = (bin_edges[1:] + bin_edges[:-1]) / 2.0
        bin_width = bin_edges[1] - bin_edges[0]

        fig = plt.figure(figsize=figsize, dpi=dpi, **kwargs)
        plt.bar(bin_middles, haar_prob, width=bin_width, label="Haar")
        plt.bar(bin_middles, pqc_prob, width=bin_width, label="PQC", alpha=0.6)
        plt.xlim((-0.05, 1.05))
        plt.ylim(bottom=0.0, top=max(max(pqc_prob), max(haar_prob)) + 0.01)
        plt.grid(True)
        plt.title(f"Expressibility: {np.round(expr,5)}")
        plt.xlabel("Fidelity")
        plt.ylabel("Probability")
        plt.legend()

        return fig
