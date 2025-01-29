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

import re
from typing import TypedDict
import logging

import networkx as nx
import numpy as np
from dimod import qubo_to_ising
from more_itertools import locate
from qiskit_optimization.applications import Tsp
from qiskit_optimization.converters import QuadraticProgramToQubo

from quark.modules.applications.Mapping import Mapping, Core
from quark.modules.applications.optimization.TSP.mappings.QUBO import QUBO
from quark.utils import start_time_measurement, end_time_measurement


class Ising(Mapping):
    """
    Ising formulation for the TSP.
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__()
        self.submodule_options = ["QAOA", "PennylaneQAOA", "QiskitQAOA"]
        self.key_mapping = None
        self.graph = None
        self.config = None

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module.

        :return: List of dict with requirements of this module
        """
        return [
            {"name": "networkx", "version": "3.4.2"},
            {"name": "numpy", "version": "1.26.4"},
            {"name": "dimod", "version": "0.12.18"},
            {"name": "more-itertools", "version": "10.5.0"},
            {"name": "qiskit-optimization", "version": "0.6.1"},
            *QUBO.get_requirements()
        ]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this mapping.

        :return: Dictionary containing parameter options.
        .. code-block:: python

            return {
                    "lagrange_factor": {
                        "values": [0.75, 1.0, 1.25],
                        "description": "By which factor would you like to multiply your lagrange?"
                    },
                    "mapping": {
                        "values": ["ocean", "qiskit"],
                        "description": "Which Ising formulation of the TSP problem should be used?"
                    }
                }
        """
        return {
            "lagrange_factor": {
                "values": [0.75, 1.0, 1.25],
                "description": "By which factor would you like to multiply your lagrange?"
            },
            "mapping": {
                "values": ["ocean", "qiskit"],
                "description": "Which Ising formulation of the TSP problem should be used?"
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config.

        .. code-block:: python

             lagrange_factor: float
             mapping: str

        """
        lagrange_factor: float
        mapping: str

    def map(self, problem: nx.Graph, config: Config) -> tuple[dict, float]:
        """
        Maps the networkx graph to an Ising formulation.

        :param problem: Networkx graph
        :param config: Config with the parameters specified in Config class
        :return: Dict with Ising, time it took to map it
        """
        self.graph = problem
        self.config = config
        # Call mapping function defined in configuration
        mapping = self.config["mapping"]
        if mapping == "ocean":
            return self._map_ocean(problem, config)
        elif mapping == "qiskit":
            return self._map_qiskit(problem, config)
        else:
            logging.error(f"Unknown mapping {mapping}.")
            raise ValueError(f"Unknown mapping {mapping}.")

    def _map_ocean(self, graph: nx.Graph, config: Config) -> tuple[dict, float]:
        """
        Use D-Wave/Ocean TSP QUBO/Ising model.

        :param graph: Networkx graph
        :param config: Config with the parameters specified in Config class
        :return: Dict with the Ising, time it took to map it
        """
        start = start_time_measurement()
        qubo_mapping = QUBO()
        q, _ = qubo_mapping.map(graph, config)
        t, j, _ = qubo_to_ising(q["Q"])

        # Convert ISING dict to matrix
        timesteps = graph.number_of_nodes()
        matrix_size = graph.number_of_nodes() * timesteps
        j_matrix = np.zeros((matrix_size, matrix_size), dtype=float)

        self.key_mapping = {}
        index_counter = 0

        for key, value in j.items():
            if key[0] not in self.key_mapping:
                self.key_mapping[key[0]] = index_counter
                index_counter += 1
            if key[1] not in self.key_mapping:
                self.key_mapping[key[1]] = index_counter
                index_counter += 1
            u = self.key_mapping[key[0]]
            v = self.key_mapping[key[1]]
            j_matrix[u][v] = value

        return {"J": j_matrix, "t": np.array(list(t.values())), "J_dict": j}, end_time_measurement(start)

    @staticmethod
    def _map_qiskit(graph: nx.Graph, config: Config) -> tuple[dict, float]:
        """
        Use Ising Mapping of Qiskit Optimize:
        TSP class: https://qiskit.org/documentation/optimization/stubs/qiskit_optimization.applications.Tsp.html
        Example notebook: https://qiskit.org/documentation/tutorials/optimization/6_examples_max_cut_and_tsp.html

        :param graph: Networkx graph
        :param config: Config with the parameters specified in Config class
        :return: Dict with the Ising, time it took to map it
        """
        start = start_time_measurement()
        tsp = Tsp(graph)
        qp = tsp.to_quadratic_program()
        logging.info(qp.export_as_lp_string())
        qp2qubo = QuadraticProgramToQubo()
        qubo = qp2qubo.convert(qp)
        qubitOp, _ = qubo.to_ising()

        # Reverse generate J and t out of qubit PauliSumOperator from qiskit
        t_matrix = np.zeros(qubitOp.num_qubits, dtype=complex)
        j_matrix = np.zeros((qubitOp.num_qubits, qubitOp.num_qubits), dtype=complex)
        pauli_list = qubitOp.to_list()

        for pauli_str, coeff in pauli_list:
            pauli_str_list = list(pauli_str)
            index_pos_list = list(locate(pauli_str_list, lambda a: a == 'Z'))
            if len(index_pos_list) == 1:
                t_matrix[index_pos_list[0]] = coeff
            elif len(index_pos_list) == 2:
                j_matrix[index_pos_list[0]][index_pos_list[1]] = coeff

        return {"J": j_matrix, "t": t_matrix}, end_time_measurement(start)

    def reverse_map(self, solution: any) -> tuple[dict, float]:
        """
        Maps the solution back to the representation needed by the TSP class for validation/evaluation.

        :param solution: List or array containing the solution
        :return: Solution mapped accordingly, time it took to map it
        """
        start = start_time_measurement()
        if -1 in solution:  # ising model output from Braket QAOA
            solution = self._convert_ising_to_qubo(solution)
        elif self.config["mapping"] == "ocean":
            logging.debug("Flip bits in solutions to unify different mappings")
            solution = self._flip_bits_in_bitstring(solution)

        logging.info(f"Best Bitstring: {solution}")
        n = self.graph.number_of_nodes()

        result = {}
        if self.key_mapping is None:
            # Node indexes in graph are used as index in qubits
            it = np.nditer(solution, flags=['multi_index'])
            for x in it:
                logging.debug(f"{x}, {it.multi_index}")
                idx = it.multi_index[0]
                result[(int(idx / n), int(idx % n))] = x
        else:
            logging.debug("Using key Mapping: {self.key_mapping}")
            for key, value in self.key_mapping.items():
                result[key] = 1 if solution[value] == 1 else 0

        return result, end_time_measurement(start)

    @staticmethod
    def _flip_bits_in_bitstring(solution: any) -> any:
        """
        Flip bits in the solution bitstring to unify different mappings.

        :param solution: Solution bitstring
        :return: Flipped solution bitstring
        """
        solution = np.array(solution)
        with np.nditer(solution, op_flags=['readwrite']) as it:
            for x in it:
                x[...] = 1 - x

        return solution

    @staticmethod
    def _convert_ising_to_qubo(solution: any) -> any:
        """
        Convert Ising model output to QUBO format.

        :param solution: Ising model output
        :return: QUBO format solution
        """
        solution = np.array(solution)
        with np.nditer(solution, op_flags=['readwrite']) as it:
            for x in it:
                if x == -1:
                    x[...] = 0

        return solution

    def get_default_submodule(self, option: str) -> Core:
        """
        Get the default submodule based on the given option.

        :param option: Submodule option
        :return: Corresponding submodule
        :raises NotImplemented: If the provided option is not implemented
        """
        if option == "QAOA":
            from quark.modules.solvers.QAOA import QAOA  # pylint: disable=C0415
            return QAOA()
        elif option == "PennylaneQAOA":
            from quark.modules.solvers.PennylaneQAOA import PennylaneQAOA  # pylint: disable=C0415
            return PennylaneQAOA()
        elif option == "QiskitQAOA":
            from quark.modules.solvers.QiskitQAOA import QiskitQAOA  # pylint: disable=C0415
            return QiskitQAOA()
        else:
            raise NotImplementedError(f"Solver Option {option} not implemented")
