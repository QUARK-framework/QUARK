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

"""Module to evaluate the entanglement spectrum of circuits."""

import typing

from qiskit_aer.noise import NoiseModel as qiskitNoiseModel

from qiskit.quantum_info import partial_trace
from scipy.spatial.distance import jensenshannon

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from ..interface.metas import MetaExplorer
from ..interface.circuit import CircuitDescriptor
from ..simulators.circuit_simulators import CircuitSimulator

NOISE_MODELS = {
    "qiskit": qiskitNoiseModel,
}


class EntanglementSpectrum(MetaExplorer):
    """Calculates entanglement spectrum of a parameterized quantum circuit"""

    def __init__(
        self,
        circuit: CircuitDescriptor,
        noise_model: typing.Union[
            qiskitNoiseModel, None
        ] = None,
        samples: int = 1000,
        tapered_indices: tuple = tuple(),
        cutoff: int = -30,
    ):
        """Constructor the the Expresssibility analyzer

        :param circuit: input circuit as a CircuitDescriptor object
        :param noise_model:  (dict, NoiseModel) initialization noise-model dictionary
        :param samples: number of samples for the experiment
        :param tapered_indices: qubits to be tapered for bipartiting the system
        :param cutoff: minimum cutoff value for the eigenvalues
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
        self.ent_spec = 0.0
        self.cutoff = cutoff
        if tapered_indices:
            if len(set(tapered_indices)) != self.circuit.num_qubits // 2:
                raise ValueError(
                    f"The provided tapered_indices must exactly have half \
                    the number of total qubits present in the system."
                )
            self.tapered_indices = tapered_indices
        else:
            self.tapered_indices = tuple(
                range(self.circuit.num_qubits // 2, self.circuit.num_qubits)
            )
        self.eigvals_sample: typing.List[float] = []
        self.plot_data: typing.List[np.ndarray] = []

    @staticmethod
    def kl_divergence(prob_a: np.ndarray, prob_b: np.ndarray) -> float:
        """Returns KL divergence between two probabilities"""
        prob_a[prob_a == 0] = 1e-10
        kl_div = np.sum(np.where(prob_a != 0, prob_a * np.log(prob_a / prob_b), 0))
        return typing.cast(float, kl_div)

    @staticmethod
    def marchenko_pastur_pdf(x, gamma):
        """Computes the probability density function (PDF) for the Marchenko-Pastur distribution"""
        lambda_plus = np.power(1 + np.sqrt(1 / gamma), 2)  # Largest eigenvalue
        lambda_minus = np.power(1 - np.sqrt(1 / gamma), 2)  # Smallest eigenvalue
        x_gamma_geq = np.maximum(lambda_plus - x, np.zeros_like(x))
        x_gamma_leq = np.maximum(x - lambda_minus, np.zeros_like(x))

        return np.sqrt(x_gamma_geq * x_gamma_leq) / (2 * np.pi * gamma * x)

    @staticmethod
    def inverse_transform_sampling(data, n_bins=1000, n_samples=1000):
        """Samples from a given distribution followed by a given set of data"""
        hist, bin_edges = np.histogram(data, bins=n_bins, density=True)
        cum_values = np.zeros(bin_edges.shape)
        cum_values[1:] = np.cumsum(hist * np.diff(bin_edges))
        return sp.interpolate.interp1d(cum_values, bin_edges)(np.random.rand(n_samples))

    def gen_params(self) -> typing.List[typing.Dict[typing.Any, typing.Any]]:
        """Generate parameters for the calculation of expressibility

        :returns theta: first list of parameters for the parameterized quantum circuit
        """
        theta = [
            {p: 2 * np.random.random() * np.pi for p in self.circuit.parameters}
            for _ in range(self.num_samples)
        ]
        return theta

    def prob_pqc(self, shots: int = 1024) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Return probability density function of fidelities for PQC

        :param shots: number of shots for circuit execution
        :returns eigvals (np.array): np.array of all eigenvalues
        :returns mean_eigvals (np.array): np.array of sample-wise mean of all eigenvalues
        """
        thetas = self.gen_params()
        theta_circuits = [
            CircuitSimulator(self.circuit, self.noise_model).simulate(theta, shots)
            for theta in thetas
        ]
        rho_circs = [
            -sp.linalg.logm(partial_trace(rho, self.tapered_indices).data)
            for rho in theta_circuits
        ]
        eigvals = [np.round(np.sort(np.linalg.eigvals(rho)), 5) for rho in rho_circs]
        mean_eigvals = -np.mean(eigvals, axis=0)
        mean_eigvals[np.where(mean_eigvals < self.cutoff)[0]] = self.cutoff
        self.eigvals_sample = mean_eigvals
        return np.array(eigvals), mean_eigvals

    def entanglement_spectrum(
        self, measure: str = "kld", shots: int = 1024
    ) -> typing.Tuple[float, np.ndarray]:
        r"""Returns entanglement spectrum divergence (ESD) for the circuit against
        Marchenko-Pastur distribution

        .. math::
            ESD = D_{KL}(\hat{P}_{PQC}(H_{\text{ent}}; \theta) | P_{Haar}(H_{\text{ent}}))\\
            ESD = D_{\sqrt{JSD}}(\hat{P}_{PQC}(H_{\text{ent}}; \theta) | P_{Haar}(H_{\text{ent}}))

        :param measure: specifies measure used in the entanglement spectrum divergence calculation
        :param shots: number of shots for circuit execution
        :returns pqc_esd: float, entanglement spectrum divergence value
        :raises ValueError: if invalid measure is specified
        """

        num_rows, num_cols = 2 ** (self.circuit.num_qubits // 2), 2 ** (
            self.circuit.num_qubits // 2
        )

        if len(self.circuit.parameters) > 0:
            eigvals, mean_eigvals = self.prob_pqc(shots)
        else:
            cor = np.corrcoef(np.random.normal(0, 1, size=(num_rows, num_cols)))
            eigvals = np.array([np.linalg.eig(cor)[0]] * 10000)
            mean_eigvals = -np.mean(eigvals, axis=0)
            mean_eigvals[np.where(mean_eigvals < self.cutoff)[0]] = self.cutoff

        gamma = 1
        x_min = np.power(1 - np.sqrt(1 / gamma), 2)
        x_min = 1e-1 if x_min < 1e-1 else x_min
        x_max = np.power(1 + np.sqrt(1 / gamma), 2)
        x = np.linspace(x_min, x_max, 1000)

        haar_prob = self.marchenko_pastur_pdf(x, gamma)
        haar_prob /= np.sum(haar_prob)

        bin_edges: np.ndarray
        pqc_hist, bin_edges = np.histogram(eigvals.real, 1000, density=True)
        pqc_prob: np.ndarray = pqc_hist / float(pqc_hist.sum())

        pqc_prob[np.where(pqc_prob == 0.0)[0]] = 1e-9
        haar_prob[np.where(haar_prob == 0.0)[0]] = 1e-9

        if measure == "kld":
            pqc_esd = self.kl_divergence(pqc_prob, haar_prob)
        elif measure == "jsd":
            pqc_esd = jensenshannon(pqc_prob, haar_prob, 2.0)
        else:
            raise ValueError("Invalid measure provided, choose from 'kld' or 'jsd'")

        mpd = np.array(self.inverse_transform_sampling(haar_prob, 10000, 256)).astype(
            float
        )
        self.plot_data = [mpd, bin_edges]
        self.ent_spec = pqc_esd

        return pqc_esd, mean_eigvals

    def plot(self, data, figsize=(6, 4), dpi=300, **kwargs):
        """Returns plot for expressibility visualization"""

        num_rows = 2 ** (self.circuit.num_qubits // 2)
        gamma = 1
        x_min = np.power(1 - np.sqrt(1 / gamma), 2)
        x_min = 1e-1 if x_min < 1e-1 else x_min
        x_max = np.power(1 + np.sqrt(1 / gamma), 2)
        x = np.linspace(x_min, x_max, 1000)

        mpd = np.array(
            self.inverse_transform_sampling(
                self.marchenko_pastur_pdf(x, gamma), 10000, num_rows
            )
        ).astype(float)
        fcol_min = np.min(np.array(data)[:, 0]).real

        ticks = np.arange(1, len(data) + 1)
        cmap = plt.get_cmap("turbo", len(data))
        norm = matplotlib.colors.BoundaryNorm(np.arange(len(data) + 1) + 0.5, len(data))
        smap = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

        fig = plt.figure(figsize=(12, 8), facecolor="white")
        for idx, layer in enumerate(data):
            plt.plot(range(len(layer)), layer.real, c=cmap(idx))

        plt.plot(
            range(len(mpd)),
            np.sort(-mpd)[::-1] + fcol_min,
            "--",
            label="Marchenko-Pastur\n distribution",
            c="black",
        )

        plt.legend(fontsize=12, loc="upper right")
        plt.ylabel(r"$\xi_k$", fontsize=16)
        plt.xlabel("k", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        # add color bar
        fig.colorbar(smap, ticks=ticks)

        return fig
