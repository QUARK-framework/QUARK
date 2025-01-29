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

import logging
from typing import TypedDict

import numpy as np

from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Sampler, Estimator
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_optimization.applications import OptimizationApplication
from qiskit_algorithms.optimizers import POWELL, SPSA, COBYLA
from qiskit_algorithms.minimum_eigensolvers import VQE, QAOA, NumPyMinimumEigensolver

from quark.modules.solvers.Solver import Solver
from quark.modules.Core import Core
from quark.utils import start_time_measurement, end_time_measurement


class QiskitQAOA(Solver):
    """
    Qiskit QAOA.
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__()
        self.submodule_options = ["qasm_simulator", "qasm_simulator_gpu"]
        self.ry = None

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module.

        :return: List of dict with requirements of this module
        """
        return [
            {"name": "qiskit", "version": "1.3.0"},
            {"name": "qiskit-optimization", "version": "0.6.1"},
            {"name": "numpy", "version": "1.26.4"},
            {"name": "qiskit-algorithms", "version": "0.3.1"}
        ]

    def get_default_submodule(self, option: str) -> Core:
        """
        Returns the default submodule based on the provided option.

        :param option: The name of the submodule
        :return: Instance of the default submodule
        """
        if option in ["qasm_simulator", "qasm_simulator_gpu"]:
            from quark.modules.devices.HelperClass import HelperClass  # pylint: disable=C0415
            return HelperClass(option)
        else:
            raise NotImplementedError(f"Device Option {option} not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this solver.

        :return: Dictionary of configurable settings
        .. code-block:: python

                    return {
                            "shots": {  # number measurements to make on circuit
                                "values": list(range(10, 500, 30)),
                                "description": "How many shots do you need?"
                            },
                            "iterations": {  # number measurements to make on circuit
                                "values": [1, 5, 10, 20, 50, 75],
                                "description": "How many iterations do you need? Warning: When using\
                                the IBM Eagle Device you should only choose a lower number of\
                                iterations, since a high number would lead to a waiting time that\
                                could take up to multiple days!"
                            },
                            "depth": {
                                "values": [2, 3, 4, 5, 10, 20],
                                "description": "How many layers for QAOA (Parameter: p) do you want?"
                            },
                            "method": {
                                "values": ["classic", "vqe", "qaoa"],
                                "description": "Which Qiskit solver should be used?"
                            },
                            "optimizer": {
                                "values": ["POWELL", "SPSA", "COBYLA"],
                                "description": "Which Qiskit solver should be used? Warning: When\
                                using the IBM Eagle Device you should not use the SPSA optimizer,\
                                since it is not suited for only one evaluation!"
                            }
                        }

        """
        return {
            "shots": {  # number measurements to make on circuit
                "values": list(range(10, 500, 30)),
                "description": "How many shots do you need?"
            },
            "iterations": {  # number measurements to make on circuit
                "values": [1, 5, 10, 20, 50, 75],
                "description": "How many iterations do you need? Warning: When using the IBM Eagle device you should "
                               "only choose a low number of iterations (long computation times)."
            },
            "depth": {
                "values": [2, 3, 4, 5, 10, 20],
                "description": "How many layers for QAOA (parameter: p) do you want?"
            },
            "method": {
                "values": ["classic", "vqe", "qaoa"],
                "description": "Which Qiskit solver should be used?"
            },
            "optimizer": {
                "values": ["POWELL", "SPSA", "COBYLA"],
                "description": "Which Qiskit solver should be used? Warning: When using the IBM Eagle device you should"
                               " not use the SPSA optimizer for a low number of iterations!"
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config.

        .. code-block:: python

            shots: int
            depth: int
            iterations: int
            layers: int
            method: str
            optimizer: str

        """
        shots: int
        depth: int
        iterations: int
        layers: int
        method: str
        optimizer: str

    @staticmethod
    def normalize_data(data: any, scale: float = 1.0) -> any:
        """
        Not used currently, as I just scale the coefficients in the qaoa_operators_from_ising.

        :param data: Data to normalize
        :param scale: Scaling factor
        :return: Normalized data
        """
        return scale * data / np.max(np.abs(data))

    def run(self, mapped_problem: any, device_wrapper: any, config: Config, **kwargs: dict) -> tuple[any, float, dict]:
        """
        Run Qiskit QAOA algorithm on Ising.

        :param mapped_problem: Dictionary with the keys 'J' and 't'
        :param device_wrapper: Instance of device
        :param config: Config object for the solver
        :param kwargs: No additionally settings needed
        :return: Solution, the time it took to compute it and optional additional information
        """
        J = mapped_problem['J']
        t = mapped_problem['t']
        start = start_time_measurement()
        ising_op = self._get_pauli_op((t, J))

        if config["method"] == "classic":
            algorithm = NumPyMinimumEigensolver()
        else:
            optimizer = None
            if config["optimizer"] == "COBYLA":
                optimizer = COBYLA(maxiter=config["iterations"])
            elif config["optimizer"] == "POWELL":
                optimizer = POWELL(maxiter=config["iterations"], maxfev=config["iterations"] if
                                   device_wrapper.device == 'ibm_eagle' else None)
            elif config["optimizer"] == "SPSA":
                optimizer = SPSA(maxiter=config["iterations"])
            if config["method"] == "vqe":
                self.ry = TwoLocal(ising_op.num_qubits, "ry", "cz", reps=config["depth"], entanglement="full")
                estimator = Estimator()
                algorithm = VQE(ansatz=self.ry, optimizer=optimizer, estimator=estimator)
            elif config["method"] == "qaoa":
                sampler = Sampler()
                algorithm = QAOA(reps=config["depth"], optimizer=optimizer, sampler=sampler)
            else:
                logging.warning("No method selected in QiskitQAOA. Continue with NumPyMinimumEigensolver.")
                algorithm = NumPyMinimumEigensolver()

        # Run actual optimization algorithm
        try:
            result = algorithm.compute_minimum_eigenvalue(ising_op)
        except ValueError as e:
            logging.error(f"The following ValueError occurred in module QiskitQAOA: {e}")
            logging.error("The benchmarking run terminates with exception.")
            raise Exception("Please refer to the logged error message.") from e

        best_bitstring = self._get_best_solution(result)
        return best_bitstring, end_time_measurement(start), {}

    def _get_best_solution(self, result) -> any:
        """
        Gets the best solution from the result.

        :param result: Result from the quantum algorithm
        :return: Best bitstring solution
        """
        if self.ry is not None:
            if hasattr(result, "optimal_point"):
                para_dict = dict(zip(self.ry.parameters, result.optimal_point))
                unbound_para = set(self.ry.parameters) - set(para_dict.keys())
                for param in unbound_para:
                    para_dict[param] = 0.0
                eigvec = Statevector(self.ry.assign_parameters(para_dict))
            elif hasattr(result, "eigenstate"):
                eigvec = result.eigenstate
            else:
                raise AttributeError("The result object does not have 'optimal_point' or 'eigenstate' attributes.")
        else:
            if hasattr(result, "eigenstate"):
                eigvec = result.eigenstate
            else:
                raise AttributeError("The result object does not have 'eigenstate'.")

        best_bitstring = OptimizationApplication.sample_most_likely(eigvec)
        return best_bitstring

    @staticmethod
    def _get_pauli_op(ising: tuple[np.ndarray, np.ndarray]) -> SparsePauliOp:
        """
        Creates a Pauli operator from the given Ising model representation.

        :param ising: Tuple with linear and quandratic terms
        .return: SparsePauliOp representing the Ising model
        """
        pauli_list = []
        number_qubits = len(ising[0])

        # Linear terms
        it = np.nditer(ising[0], flags=['multi_index'])
        for x in it:
            logging.debug(f"{x},{it.multi_index}")
            key = it.multi_index[0]
            pauli_str = "I" * number_qubits
            pauli_str_list = list(pauli_str)
            pauli_str_list[key] = "Z"
            pauli_str = "".join(pauli_str_list)
            pauli_list.append((pauli_str, complex(x)))

        # J Part with 2-Qubit interactions
        it = np.nditer(ising[1], flags=['multi_index'])
        for x in it:
            logging.debug(f"{x},{it.multi_index}")
            idx1 = it.multi_index[0]
            idx2 = it.multi_index[1]
            pauli_str = "I" * number_qubits
            pauli_str_list = list(pauli_str)
            pauli_str_list[idx1] = "Z"
            pauli_str_list[idx2] = "Z"
            pauli_str = "".join(pauli_str_list)
            pauli_list.append((pauli_str, complex(x)))

        ising_op = SparsePauliOp.from_list(pauli_list)
        return ising_op
