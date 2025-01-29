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

from typing import TypedDict
import pickle
import logging
import os

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from quark.modules.applications.Application import Core
from quark.modules.applications.optimization.Optimization import Optimization
from quark.utils import start_time_measurement, end_time_measurement


class TSP(Optimization):
    """
    "The famous travelling salesman problem (also called the travelling salesperson problem or in short TSP) is a
    well-known NP-hard problem in combinatorial optimization, asking for the shortest possible route that visits each
    node exactly once, given a list of nodes and the distances between each pair of nodes. Applications of the
    TSP can be found in planning, logistics, and the manufacture of microchips. In these applications, the general
    concept of a node represents, for example, customers, or points on a chip.

    TSP as graph problem: The solution to the TSP can be viewed as a specific ordering of the vertices in a weighted
    graph. Taking an undirected weighted graph, nodes correspond to the graph's nodes, with paths corresponding to the
    graph's edges, and a path's distance is the edge's weight."
    (source: https://github.com/aws/amazon-braket-examples/tree/main/examples)
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__("TSP")
        self.submodule_options = [
            "Ising", "QUBO", "GreedyClassicalTSP", "ReverseGreedyClassicalTSP", "RandomTSP"
        ]

    @staticmethod
    def get_requirements() -> list:
        """
        Return requirements of this module.

        :return: List of dict with requirements of this module
        """
        return [
            {"name": "networkx", "version": "3.4.2"},
            {"name": "numpy", "version": "1.26.4"}
        ]

    def get_solution_quality_unit(self) -> str:
        """
        Returns the unit of measurement for the solution quality.

        :return: Unit of measurement for the solution quality
        """
        return "Tour cost"

    def get_default_submodule(self, option: str) -> Core:
        """
        Returns the default submodule based on the given option.

        :param option: The chosen submodule option
        :return: The corresponding submodule instance
        :raises NotImplemented: If the provided option is not implemented
        """
        if option == "Ising":
            from quark.modules.applications.optimization.TSP.mappings.ISING import Ising  # pylint: disable=C0415
            return Ising()
        elif option == "QUBO":
            from quark.modules.applications.optimization.TSP.mappings.QUBO import QUBO  # pylint: disable=C0415
            return QUBO()
        elif option == "GreedyClassicalTSP":
            from quark.modules.solvers.GreedyClassicalTSP import GreedyClassicalTSP  # pylint: disable=C0415
            return GreedyClassicalTSP()
        elif option == "ReverseGreedyClassicalTSP":
            from quark.modules.solvers.ReverseGreedyClassicalTSP import ReverseGreedyClassicalTSP  # pylint: disable=C0415
            return ReverseGreedyClassicalTSP()
        elif option == "RandomTSP":
            from quark.modules.solvers.RandomClassicalTSP import RandomTSP  # pylint: disable=C0415
            return RandomTSP()
        else:
            raise NotImplementedError(f"Mapping Option {option} not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this application

        :return: Dictionary with configurable settings.
        .. code-block:: python

            return {
                    "nodes": {
                        "values": list([3, 4, 6, 8, 10, 14, 16]),
                        "allow_ranges": True,
                        "description": "How many nodes does your graph need?",
                        "postproc": int
                    }
                }
        """
        return {
            "nodes": {
                "values": list([3, 4, 6, 8, 10, 14, 16]),
                "allow_ranges": True,
                "description": "How many nodes does you graph need?",
                "postproc": int  # postproc needed to parse the result from allow_ranges to int
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config.

        .. code-block:: python

             nodes: int

        """
        nodes: int

    @staticmethod
    def _get_tsp_matrix(graph: nx.Graph) -> np.ndarray:
        """
        Creates distance matrix out of given coordinates.

        :param graph: The input graph
        :return: Distance matrix
        """
        number_of_nodes = len(graph)
        matrix = np.zeros((number_of_nodes, number_of_nodes))
        for i in nx.all_pairs_dijkstra_path_length(graph, weight="weight"):
            distance_dist = i[1]
            for j in distance_dist.items():
                matrix[i[0] - 1][j[0] - 1] = j[1]
                matrix[j[0] - 1][i[0] - 1] = matrix[i[0] - 1][j[0] - 1]

        return matrix

    def generate_problem(self, config: Config) -> nx.Graph:
        """
        Uses the reference graph to generate a problem for a given config.

        :param config: Configuration dictionary
        :return: Graph with the problem
        """

        if config is None:
            config = {"nodes": 5}

        nodes = config['nodes']

        # Read in the original graph
        with open(os.path.join(os.path.dirname(__file__), "data", "reference_graph.gpickle"), "rb") as file:
            graph = pickle.load(file)

        # Remove seams until the target number of seams is reached
        nodes_in_graph = list(graph.nodes)
        nodes_in_graph.sort()

        if len(nodes_in_graph) < nodes:
            raise ValueError("Too many nodes! The original graph has less seams than that!")

        unwanted_nodes = nodes_in_graph[-len(nodes_in_graph) + nodes:]
        unwanted_nodes = [x for x in graph.nodes if x in unwanted_nodes]

        # Remove one node after another
        for node in unwanted_nodes:
            graph.remove_node(node)

        if not nx.is_connected(graph):
            logging.error("Graph is not connected!")
            raise ValueError("Graph is not connected!")

        # Normalize graph
        cost_matrix = self._get_tsp_matrix(graph)
        graph = nx.from_numpy_array(cost_matrix)

        self.application = graph

        return graph

    def process_solution(self, solution: dict) -> tuple[list, float]:
        """
        Convert dict to list of visited nodes.

        :param solution: Dictionary with solution
        :return: Processed solution and the time it took to process it
        """
        start_time = start_time_measurement()
        nodes = self.application.nodes()
        start = np.min(nodes)
        # fill route with None values
        route: list = [None] * len(self.application)

        # Get nodes from sample
        logging.info(str(solution.items()))

        for (node, timestep), val in solution.items():
            if val:
                logging.info((node, timestep))
            if val and (node not in route):
                route[timestep] = node

        # Check whether every timestep has only 1 node flagged
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
                return None, end_time_measurement(start_time)

        # Check validity of solution
        if sum(value == 1 for value in solution.values()) > len(route):
            logging.warning("Result is longer than route! This might be problematic!")
            return None, end_time_measurement(start_time)

        # Run heuristic replacing None values
        if None in route:
            # get not assigned nodes
            nodes_unassigned = [node for node in list(nodes) if node not in route]
            nodes_unassigned = list(np.random.permutation(nodes_unassigned))
            for idx, node in enumerate(route):
                if node is None:
                    route[idx] = nodes_unassigned[0]
                    nodes_unassigned.remove(route[idx])

        # Cycle solution to start at provided start location
        if start is not None and route[0] != start:
            # Rotate to put the start in front
            idx = route.index(start)
            route = route[idx:] + route[:idx]

        # Log route
        parsed_route = ' ->\n'.join([f' Node {visit}' for visit in route])
        logging.info(f"Route found:\n{parsed_route}")

        return route, end_time_measurement(start_time)

    def validate(self, solution: list) -> tuple[bool, float]:
        """
        Checks if it is a valid TSP tour.

        :param solution: List containing the nodes of the solution
        :return: Boolean whether the solution is valid, time it took to validate
        """
        start = start_time_measurement()
        nodes = self.application.nodes()

        if solution is None:
            return False, end_time_measurement(start)
        elif len([node for node in list(nodes) if node not in solution]) == 0:
            logging.info(f"All {len(solution)} nodes got visited")
            return True, end_time_measurement(start)
        else:
            logging.error(f"{len([node for node in list(nodes) if node not in solution])} nodes were NOT visited")
            return False, end_time_measurement(start)

    def evaluate(self, solution: list) -> tuple[float, float]:
        """
        Find distance for given route and original data.

        :param solution: List containing the nodes of the solution
        :return: Tour cost and the time it took to calculate it
        """
        start = start_time_measurement()
        # Get the total distance without return
        total_dist = 0
        for idx, _ in enumerate(solution[:-1]):
            dist = self.application[solution[idx + 1]][solution[idx]]
            total_dist += dist['weight']

        logging.info(f"Total distance (without return): {total_dist}")

        # Add distance between start and end point to complete cycle
        return_distance = self.application[solution[0]][solution[-1]]['weight']

        # Get distance for full cycle
        distance_with_return = total_dist + return_distance
        logging.info(f"Total distance (including return): {distance_with_return}")

        return distance_with_return, end_time_measurement(start)

    def save(self, path: str, iter_count: int) -> None:
        """
        Save the current application state to a file.

        :param path: The directory path where the file will be saved
        :param iter_count: The iteration count to include in the filename
        """
        with open(f"{path}/graph_iter_{iter_count}.gpickle", "wb") as file:
            pickle.dump(self.application, file, pickle.HIGHEST_PROTOCOL)

    def visualize_solution(self, processed_solution: list[int], path: str):
        """
        Plot a graph representing the problem network with the solution path highlighted

        :param processed_solution: The solution already processed by :func:`process_solution`, a list of visited node IDs in order of being visited.
        :param path: File path for the plot
        :returns: None
        """
        NODE_SIZE = 300   # Default=300
        EDGE_WIDTH = 1.0  # Default=1.0
        FONT_SIZE = 12    # Default=12

        path_edges = list(nx.utils.pairwise(processed_solution, cyclic=True))
        path_edges = [(u, v) if u < v else (v, u) for (u, v) in path_edges]
        G = self.application
        pos = nx.circular_layout(G)
        weights = nx.get_edge_attributes(G, "weight")
        filtered_weights = {e: (int(weights[e])) for e in path_edges}

        nx.draw_networkx_nodes(G, pos, node_size=NODE_SIZE)
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=EDGE_WIDTH, edge_color="gray")
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=2 * EDGE_WIDTH, edge_color="red", arrows=True)
        nx.draw_networkx_labels(G, pos, font_size=FONT_SIZE)
        nx.draw_networkx_edge_labels(G, pos, filtered_weights, font_size=.5 * FONT_SIZE)

        plt.savefig(path)
        plt.close()
