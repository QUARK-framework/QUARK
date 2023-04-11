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
import re
from typing import TypedDict, Union

import networkx
import networkx as nx
import numpy as np
from dimod import qubo_to_ising
from more_itertools import locate
from pyqubo import Array, Placeholder, Constraint
from qiskit_optimization.applications import Tsp
from qiskit_optimization.converters import QuadraticProgramToQubo

from applications.Mapping import *
from applications.TSP.mappings.QUBO import Qubo
from solvers.PennylaneQAOA import PennylaneQAOA
from solvers.QAOA import QAOA
from solvers.QiskitQAOA import QiskitQAOA


class Ising(Mapping):
    """
    Ising formulation for the TSP.
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.solver_options = ["QAOA", "PennylaneQAOA", "QiskitQAOA"]
        self.key_mapping = None
        self.graph = None
        self.config = None

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this mapping

        :return:
                 .. code-block:: python

                     return {
                                "lagrange_factor": {
                                    "values": [0.75, 1.0, 1.25],
                                    "description": "By which factor would you like to multiply your lagrange?"
                                },
                                "mapping": {
                                    "values": ["ocean", "qiskit", "pyqubo"],
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
                "values": ["ocean", "qiskit", "pyqubo"],
                "description": "Which Ising formulation of the TSP problem should be used?"
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python

             lagrange_factor: float
             mapping: str

        """
        lagrange_factor: float
        mapping: str

    def map(self, graph: networkx.Graph, config: Config) -> (dict, float):
        """
        Maps the networkx graph to an Ising formulation.

        :param graph: networkx graph
        :type graph: networkx.Graph
        :param config: config with the parameters specified in Config class
        :param config: Config
        :return: dict with Ising, time it took to map it
        :rtype: tuple(dict, float)
        """
        self.graph = graph
        self.config = config
        # call mapping function defined in configuration
        if self.config["mapping"] == "ocean":
            return self._map_ocean(graph, config)
        elif self.config["mapping"] == "pyqubo":
            return self._map_pyqubo(graph, config)
        elif self.config["mapping"] == "qiskit":
            return self._map_qiskit(graph, config)

    @staticmethod
    def _create_pyqubo_model(cost_matrix: list) -> any:
        """
        This PyQubo formulation of the TSP was kindly provided by AWS.

        :param cost_matrix:
        :type cost_matrix: list
        :return:
        :rtype: any
        """
        n = len(cost_matrix)
        x = Array.create('c', (n, n), 'BINARY')

        # Constraint not to visit more than two nodes at the same time.
        time_const = 0.0
        for i in range(n):
            # If you wrap the hamiltonian by Const(...), this part is recognized as constraint
            time_const += Constraint((sum(x[i, j] for j in range(n)) - 1) ** 2, label="time{}".format(i))

        # Constraint not to visit the same location more than twice.
        location_const = 0.0
        for j in range(n):
            location_const += Constraint((sum(x[i, j] for i in range(n)) - 1) ** 2, label="location{}".format(j))

        # distance of route
        distance = 0.0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    d_ij = cost_matrix[i][j]
                    distance += d_ij * x[k, i] * x[(k + 1) % n, j]

        # Construct hamiltonian
        A = Placeholder("A")
        H = distance + A * (time_const + location_const)

        # Compile model
        model = H.compile()
        return model

    @staticmethod
    def _path_cost_nx(graph: networkx.Graph, route: list, weight: str = "weight") -> any:
        route_edges = [(x, y) for x, y in zip(route, route[1:])]
        sg = graph.edge_subgraph(route_edges)
        cost = sg.size(weight=weight)
        return cost

    @staticmethod
    def _get_matrix_index(ising_index_string: any, number_nodes: any) -> any:
        """
        Converts dictionary index (e.g. 'c[0][2]') in PyQubo to matrix index.

        (e.g. 2
        {('c[0][2]', 'c[2][1]'): 0.06161479507592913,
        ('c[0][0]', 'c[0][1]'): 20.0,
        ('c[1][0]', 'c[2][1]'): 0.720033199087941,
        ... }

        :param ising_index_string:
        :type ising_index_string: any
        :param number_nodes:
        :type number_nodes: any
        :return:
        :rtype: any
        """
        x = 0
        y = 0
        match = re.findall(r'(?<=\[)[0-9]*(?=\])', ising_index_string, re.S)
        if len(match) == 2:
            x = int(match[0])
            y = int(match[1])

        idx = x * number_nodes + y

        return idx

    def _map_pyqubo(self, graph: networkx.Graph, config: Config) -> (dict, float):
        """
        Use Qubo / Ising model defined in PyQubo.

        :param graph: networkx graph
        :type graph: networkx.Graph
        :param config: config with the parameters specified in Config class
        :type config: Config
        :return: dict with the Ising, time it took to map it
        :rtype: tuple(dict, float)
        """
        start = time() * 1000
        # cost_matrix = nx.to_numpy_matrix(graph) #self.get_tsp_matrix(graph)
        # cost_matrix = self.get_tsp_matrix(graph)
        cost_matrix = np.array(nx.to_numpy_matrix(graph, weight="weight"))
        model = self._create_pyqubo_model(cost_matrix)
        feed_dict = {'A': 2.0}
        if "lagrange_factor" in config:
            feed_dict = {'A': config["lagrange_factor"]}

        linear, quad, offset = model.to_ising(feed_dict=feed_dict)
        qubo, _ = model.to_qubo(feed_dict=feed_dict)

        timesteps = graph.number_of_nodes()

        t_matrix = np.array([0.0] * graph.number_of_nodes() * graph.number_of_nodes())

        for key, value in linear.items():
            idx = self._get_matrix_index(key, graph.number_of_nodes())
            t_matrix[idx] = value

        j_matrix = np.array(
            [[0.0] * graph.number_of_nodes() * timesteps for i in range(graph.number_of_nodes() * timesteps)])

        index_counter = 0
        for key, value in quad.items():
            x = self._get_matrix_index(key[0], graph.number_of_nodes())
            y = self._get_matrix_index(key[1], graph.number_of_nodes())
            j_matrix[x][y] = value
        # j_matrix = np.triu(j_matrix, k=1).astype(np.float64)
        # print("Number items in Ising dict: {} Number of non-zero entries in matrix: {}".\
        #       format(len(quad.items()), len(j_matrix.nonzero())))

        return {"J": j_matrix, "J_dict": quad, "t_dict": linear, "t": t_matrix}, round(time() * 1000 - start, 3)

    def _map_ocean(self, graph: networkx.Graph, config: Config) -> (dict, float):
        """
        Use D-Wave/Ocean TSP QUBO/Ising model:
        https://docs.ocean.dwavesys.com/en/stable/docs_dnx/reference/algorithms/generated/dwave_networkx.algorithms.tsp.traveling_salesperson_qubo.html#dwave_networkx.algorithms.tsp.traveling_salesperson_qubo

        :param graph: networkx graph
        :type graph: networkx.Graph
        :param config: config with the parameters specified in Config class
        :param config: Config
        :return: dict with the Ising, time it took to map it
        :rtype: tuple(dict, float)
        """

        start = time() * 1000
        qubo_mapping = Qubo()
        q, _ = qubo_mapping.map(graph, config)
        t, j, _ = qubo_to_ising(q["Q"])

        # Convert ISING dict to matrix
        timesteps = graph.number_of_nodes()

        number_of_edges = graph.number_of_edges()
        # timesteps = len(Q)/number_of_edges
        j_matrix = np.array(
            [[0.0] * graph.number_of_nodes() * timesteps for i in range(graph.number_of_nodes() * timesteps)])

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

        logging.info(j_matrix)
        logging.info(j_matrix.shape)
        # j_matrix = self.convert_to_upper_triangular_form(j_matrix)
        # logging.info("Upper triangle form: ")
        # logging.info(j_matrix)

        return {"J": j_matrix, "t": np.array(list(t.values())), "J_dict": j}, round(time() * 1000 - start, 3)

    @staticmethod
    def _map_qiskit(graph: networkx.Graph, config: Config) -> (dict, float):
        """
        Use Ising Mapping of Qiskit Optimize:
        TSP class: https://qiskit.org/documentation/optimization/stubs/qiskit_optimization.applications.Tsp.html
        Example notebook: https://qiskit.org/documentation/tutorials/optimization/6_examples_max_cut_and_tsp.html

        :param graph: networkx graph
        :type graph: networkx.Graph
        :param config: config with the parameters specified in Config class
        :type config: Config
        :return: dict with the Ising, time it took to map it
        :rtype: tuple(dict, float)
        """
        start = time() * 1000
        tsp = Tsp(graph)
        qp = tsp.to_quadratic_program()
        logging.info(qp.export_as_lp_string())
        qp2qubo = QuadraticProgramToQubo()
        qubo = qp2qubo.convert(qp)
        qubitOp, offset = qubo.to_ising()
        # reverse generate J and t out of qubit PauliSumOperator from qiskit
        t_matrix = np.array([0.0 + 0j] * qubitOp.num_qubits)

        j_matrix = np.array(
            [[0.0 + 0j] * qubitOp.num_qubits for i in range(qubitOp.num_qubits)])
        logging.info(j_matrix)

        for i in qubitOp:
            pauli_str, coeff = i.primitive.to_list()[0]
            logging.info((pauli_str, coeff))
            pauli_str_list = list(pauli_str)
            index_pos_list = list(locate(pauli_str_list, lambda a: a == 'Z'))
            matrix_to_process = None
            if len(index_pos_list) == 1:
                # update t
                t_matrix[index_pos_list[0]] = coeff
            elif len(index_pos_list) == 2:
                j_matrix[index_pos_list[0]][index_pos_list[1]] = coeff

        return {"J": j_matrix, "t": t_matrix}, round(time() * 1000 - start, 3)

    def reverse_map(self, solution: dict) -> (dict, float):
        """
        Maps the solution back to the representation needed by the TSP class for validation/evaluation.

        :param solution: dictionary containing the solution
        :type solution: dict
        :return: solution mapped accordingly, time it took to map it
        :rtype: tuple(dict, float)
        """
        start = time() * 1000
        if np.any(solution == "-1"):  # ising model output from Braket QAOA
            solution = self._convert_ising_to_qubo(solution)
        elif self.config["mapping"] == "pyqubo" or self.config["mapping"] == "ocean":
            logging.debug("Flip bits in solutions to unify different mappings")
            solution = self._flip_bits_in_bitstring(solution)

        logging.info(f"Best Bitstring: {solution}")
        n = self.graph.number_of_nodes()

        result = {}
        if self.key_mapping is None:
            # Node indexes in graph are used as index in qubits
            it = np.nditer(solution, flags=['multi_index'])
            for x in it:
                logging.debug("{:.2f} <{} >".format(x, it.multi_index))
                idx = it.multi_index[0]
                result[(int(idx / n), int(idx % n))] = x
        else:
            logging.debug("Using key Mapping: {self.key_mapping}")
            for key, value in self.key_mapping.items():
                result[key] = 1 if solution[value] == 1 else 0

        logging.info(result)
        return result, round(time() * 1000 - start, 3)

    @staticmethod
    def _flip_bits_in_bitstring(solution: any) -> any:
        # depending on used solver 0 or 1 can indicate a node per timestep
        # if np.count_nonzero(solution) > n:
        solution = np.array(solution)
        with np.nditer(solution, op_flags=['readwrite']) as it:
            for x in it:
                x[...] = 1 - x
        return solution

    @staticmethod
    def _convert_ising_to_qubo(solution: any) -> any:
        solution = np.array(solution)
        with np.nditer(solution, op_flags=['readwrite']) as it:
            for x in it:
                if x == -1:
                    x = 0
        return solution

    def get_solver(self, solver_option: str) -> Union[QAOA, PennylaneQAOA, QiskitQAOA]:

        if solver_option == "QAOA":
            return QAOA()
        elif solver_option == "PennylaneQAOA":
            return PennylaneQAOA()
        elif solver_option == "QiskitQAOA":
            return QiskitQAOA()
        else:
            raise NotImplementedError(f"Solver Option {solver_option} not implemented")
