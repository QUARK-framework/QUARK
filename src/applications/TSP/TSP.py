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
import os
from typing import TypedDict, Union
from time import time

import networkx
import networkx as nx
import numpy as np

from applications.Application import *
from applications.TSP.mappings.Direct import Direct
from applications.TSP.mappings.ISING import Ising
from applications.TSP.mappings.QUBO import Qubo


class TSP(Application):
    """
    \"The famous travelling salesman problem (also called the travelling salesperson problem or in short TSP) is a
    well-known NP-hard problem in combinatorial optimization, asking for the shortest possible route that visits each
    node exactly once, given a list of nodes and the distances between each pair of nodes. Applications of the
    TSP can be found in planning, logistics, and the manufacture of microchips. In these applications, the general
    concept of a node represents, for example, customers, or points on a chip.

    TSP as graph problem: The solution to the TSP can be viewed as a specific ordering of the vertices in a weighted
    graph. Taking an undirected weighted graph, nodes correspond to the graph's nodes, with paths corresponding to the
    graph's edges, and a path's distance is the edge's weight. Typically, the graph is complete where each pair of nodes
    is connected by an edge. If no connection exists between two nodes, one can add an arbitrarily long edge to complete
    the graph without affecting the optimal tour.\"
    (source: https://github.com/aws/amazon-braket-examples/blob/main/examples/quantum_annealing/Dwave_TravelingSalesmanProblem/Dwave_TravelingSalesmanProblem.ipynb)
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__("TSP")
        self.mapping_options = ["Ising", "Qubo", "Direct"]

    def get_solution_quality_unit(self) -> str:
        return "Tour cost"

    def get_mapping(self, mapping_option: str) -> Union[Ising, Qubo, Direct]:
        if mapping_option == "Ising":
            return Ising()
        elif mapping_option == "Qubo":
            return Qubo()
        elif mapping_option == "Direct":
            return Direct()
        else:
            raise NotImplementedError(f"Mapping Option {mapping_option} not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this application

        :return:
                 .. code-block:: python

                      return {
                                "nodes": {
                                    "values": list([3, 4, 6, 8, 10, 14, 16]),
                                    "description": "How many nodes does your graph need?"
                                }
                            }

        """
        return {
            "nodes": {
                "values": list([3, 4, 6, 8, 10, 14, 16]),
                "description": "How many nodes does you graph need?"
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python

             nodes: int

        """
        nodes: int

    @staticmethod
    def _get_tsp_matrix(graph: networkx.Graph) -> np.ndarray:
        """
        Creates distance matrix out of given coordinates.

        :param graph:
        :type graph: networkx.Graph
        :return:
        :rtype: np.ndarray
        """
        number_of_nodes = len(graph)
        matrix = np.zeros((number_of_nodes, number_of_nodes))
        for i in nx.all_pairs_dijkstra_path_length(graph, weight="weight"):
            distance_dist = i[1]
            for j in distance_dist.items():
                matrix[i[0] - 1][j[0] - 1] = j[1]
                matrix[j[0] - 1][i[0] - 1] = matrix[i[0] - 1][j[0] - 1]
        return matrix

    def generate_problem(self, config: Config, iter_count: int) -> networkx.Graph:
        """
        Uses the reference graph to generate a problem for a given config.

        :param config:
        :type config: Config
        :param iter_count: the iteration count
        :type iter_count: int
        :return: graph with the problem
        :rtype: networkx.Graph
        """

        if config is None:
            config = {"nodes": 5}

        nodes = config['nodes']

        # Read in the original graph
        graph = nx.read_gpickle(os.path.join(os.path.dirname(__file__), "reference_graph.gpickle"))

        # Remove seams until the target number of seams is reached
        # Get number of seam in graph
        nodes_in_graph = [x for x in graph.nodes]
        nodes_in_graph.sort()

        if len(nodes_in_graph) < nodes:
            raise ValueError(f"Too many nodes! The original graph has less seams than that!")

        unwanted_nodes = nodes_in_graph[-len(nodes_in_graph) + nodes:]
        unwanted_nodes = [x for x in graph.nodes if x in unwanted_nodes]
        # Remove one node after another
        for node in unwanted_nodes:
            graph.remove_node(node)

        if not nx.is_connected(graph):
            logging.error("Graph is not connected!")
            raise ValueError(f"Graph is not connected!")

        # normalize graph
        cost_matrix = self._get_tsp_matrix(graph)
        graph = nx.from_numpy_array(cost_matrix)

        self.application = graph
        return graph

    def process_solution(self, solution: dict) -> (list, float):
        """
        Convert dict to list of visited nodes.

        :param solution:
        :type solution: dict
        :return: processed solution and the time it took to process it
        :rtype: tuple(list, float)
        """
        start_time = time() * 1000
        nodes = self.application.nodes()
        start = np.min(nodes)
        # fill route with None values
        route = [None] * len(self.application)
        # get nodes from sample
        # NOTE: Prevent duplicate node entries by enforcing only one occurrence per node along route
        logging.info(str(solution.items()))

        for (node, timestep), val in solution.items():
            if val:
                logging.info((node, timestep))
            if val and (node not in route):
                route[timestep] = node

        # check whether every timestep has only 1 node flagged
        for i in nodes:
            relevant_nodes = []
            relevant_timesteps = []
            for (node, timestep) in solution.keys():
                if node == i:
                    relevant_nodes.append(solution[(node, timestep)])
                if timestep == i:
                    relevant_timesteps.append(solution[(node, timestep)])
            if sum(relevant_nodes) != 1 or sum(relevant_timesteps) != 1:
                # timestep or nodes have more than 1 or 0 flags
                return None, round(time() * 1000 - start_time, 3)

        # check validity of solution
        if sum(value == 1 for value in solution.values()) > len(route):
            logging.warning("Result is longer than route! This might be problematic!")
            return None, round(time() * 1000 - start_time, 3)

        # run heuristic replacing None values
        if None in route:
            # get not assigned nodes
            nodes_unassigned = [node for node in list(nodes) if node not in route]
            nodes_unassigned = list(np.random.permutation(nodes_unassigned))
            for idx, node in enumerate(route):
                if node is None:
                    route[idx] = nodes_unassigned[0]
                    nodes_unassigned.remove(route[idx])

        # cycle solution to start at provided start location
        if start is not None and route[0] != start:
            # rotate to put the start in front
            idx = route.index(start)
            route = route[idx:] + route[:idx]

        # print route
        parsed_route = ' ->\n'.join([f' Node {visit}' for visit in route])
        logging.info(f"Route found:\n{parsed_route}")
        return route, round(time() * 1000 - start_time, 3)

    def validate(self, solution: list) -> (bool, float):
        """
        Checks if it is a valid TSP tour.

        :param solution: list containing the nodes of the solution
        :type solution: list
        :return: Boolean whether the solution is valid, time it took to validate
        :rtype: tuple(bool, float)
        """
        start = time() * 1000
        nodes = self.application.nodes()

        if solution is None:
            return False, round(time() * 1000 - start, 3)
        elif len([node for node in list(nodes) if node not in solution]) == 0:
            logging.info(f"All {len(solution)} nodes got visited")
            return True, round(time() * 1000 - start, 3)
        else:
            logging.error(f"{len([node for node in list(nodes) if node not in solution])} nodes were NOT visited")
            return False, round(time() * 1000 - start, 3)

    def evaluate(self, solution: list) -> (float, float):
        """
        Find distance for given route e.g. [0, 4, 3, 1, 2] and original data.

        :param solution:
        :type solution: list
        :return: Tour cost and the time it took to calculate it
        :rtype: tuple(float, float)
        """
        start = time() * 1000
        # get the total distance without return
        total_dist = 0
        for idx, node in enumerate(solution[:-1]):
            dist = self.application[solution[idx + 1]][solution[idx]]
            total_dist += dist['weight']

        logging.info(f"Total distance (without return): {total_dist}")

        # add distance between start and end point to complete cycle
        return_distance = self.application[solution[0]][solution[-1]]['weight']
        # logging.info('Distance between start and end: ' + return_distance)

        # get distance for full cycle
        distance_with_return = total_dist + return_distance
        logging.info(f"Total distance (including return): {distance_with_return}")

        return distance_with_return, round(time() * 1000 - start, 3)

    def save(self, path: str, iter_count: int) -> None:
        nx.write_gpickle(self.application, f"{path}/graph.gpickle")
