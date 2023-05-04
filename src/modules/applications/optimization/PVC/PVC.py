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

import itertools
import os
from typing import TypedDict

import networkx as nx
import numpy as np

from modules.applications.Application import *
from modules.applications.optimization.Optimization import Optimization


class PVC(Optimization):
    """
    In modern vehicle manufacturing, robots
    take on a significant workload, including performing welding
    jobs, sealing welding joints, or applying paint to the car body.
    While the robot’s tasks vary widely, the objective remains
    the same: Perform a job with the highest possible quality in the
    shortest amount of time. For instance, to protect a car’s underbody
    from corrosion, exposed welding seams are sealed by applying
    a polyvinyl chloride layer (PVC). The welding seams need to be
    traversed by a robot to apply the material.
    It is related to TSP, but different and even more complex in some
    aspects.
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__("PVC")
        self.submodule_options = ["Ising", "Qubo", "GreedyClassicalPVC", "ReverseGreedyClassicalPVC", "RandomPVC"]

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module

        :return: list of dict with requirements of this module
        :rtype: list[dict]
        """
        return [
            {
                "name": "networkx",
                "version": "2.8.8"
            },
            {
                "name": "numpy",
                "version": "1.24.1"
            }
        ]

    def get_solution_quality_unit(self) -> str:
        return "Tour cost"

    def get_default_submodule(self, option: str) -> Core:

        if option == "Ising":
            from modules.applications.optimization.PVC.mappings.ISING import Ising  # pylint: disable=C0415
            return Ising()
        elif option == "Qubo":
            from modules.applications.optimization.PVC.mappings.QUBO import Qubo  # pylint: disable=C0415
            return Qubo()
        elif option == "GreedyClassicalPVC":
            from modules.solvers.GreedyClassicalPVC import GreedyClassicalPVC  # pylint: disable=C0415
            return GreedyClassicalPVC()
        elif option == "ReverseGreedyClassicalPVC":
            from modules.solvers.ReverseGreedyClassicalPVC import ReverseGreedyClassicalPVC  # pylint: disable=C0415
            return ReverseGreedyClassicalPVC()
        elif option == "RandomPVC":
            from modules.solvers.RandomClassicalPVC import RandomPVC  # pylint: disable=C0415
            return RandomPVC()
        else:
            raise NotImplementedError(f"Mapping Option {option} not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this application

        :return:
                 .. code-block:: python

                      return {
                                "seams": {
                                    "values": list(range(1, 18)),
                                    # Currently the graph can only be as large as the reference input graph
                                    "description": "How many seams does your graph need?"
                                }
                            }

        """
        return {
            "seams": {
                "values": list(range(1, 18)),
                # In the current implementation the graph can only be as large as the reference input graph
                "description": "How many seams does you graph need?"
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python

            seams: int

        """
        seams: int

    def generate_problem(self, config: Config) -> nx.Graph:
        """
        Uses the reference graph to generate a problem for a given config.

        :param config: Config specifying the number of seams for the problem
        :type config: Config
        :return: networkx graph representing the problem
        :rtype: networkx.Graph
        """

        if config is None:
            config = {"seams": 3}
        seams = config['seams']

        # Read in the original graph
        graph = nx.read_gpickle(os.path.join(os.path.dirname(__file__), "reference_graph.gpickle"))

        # Remove seams until the target number of seams is reached
        # Get number of seam in graph
        seams_in_graph = list({x[0] for x in graph.nodes})
        seams_in_graph.sort()
        # Remove 0 as we always need the base node 0 (which is not a seam anyway)
        seams_in_graph.remove(0)

        if len(seams_in_graph) < seams:
            raise ValueError("Too many seams! The original graph has less seams than that!")

        unwanted_seams = seams_in_graph[-len(seams_in_graph) + seams:]
        unwanted_nodes = [x for x in graph.nodes if x[0] in unwanted_seams]
        # Remove one node after another
        for node in unwanted_nodes:
            graph.remove_node(node)

        if not nx.is_strongly_connected(graph):
            logging.error("Graph is not connected!")
            raise ValueError("Graph is not connected!")

        config = [x[2]['c_start'] for x in graph.edges(data=True)]
        config = list(set(config + [x[2]['c_end'] for x in graph.edges(data=True)]))

        tool = [x[2]['t_start'] for x in graph.edges(data=True)]
        tool = list(set(tool + [x[2]['t_end'] for x in graph.edges(data=True)]))

        # Now lets fill the rest of the missing edges with high values
        # get current edges
        current_edges = [(edge[0], edge[1], edge[2]['t_start'], edge[2]['t_end'], edge[2]['c_start'], edge[2]['c_end'])
                         for edge in graph.edges(data=True)]

        # get all possible edges
        all_possible_edges = list(itertools.product(list(graph.nodes), repeat=2))
        all_possible_edges = [(edges[0], edges[1], t_start, t_end, c_start, c_end) for edges in all_possible_edges for
                              c_end in config for c_start in config for t_end in tool
                              for t_start in tool if edges[0] != edges[1]]
        # calculate missing edges
        missing_edges = [item for item in all_possible_edges if item not in current_edges]
        # add these edges with very high values
        for edge in missing_edges:
            weight = 100000  # TODO Check if this value is fine
            # c_start, t_start, c_end, t_end
            graph.add_edge(edge[0], edge[1], c_start=edge[4], t_start=edge[2], c_end=edge[5], t_end=edge[3],
                           weight=weight)

        logging.info("Created PVC problem with the following attributes:")
        logging.info(f" - Number of seams: {seams}")
        logging.info(f" - Number of different configs: {len(config)}")
        logging.info(f" - Number of different tools: {len(tool)}")

        self.application = graph
        return graph.copy()

    def process_solution(self, solution: dict) -> (list, bool):
        """
        Convert dict to list of visited seams.

        :param solution: unprocessed solution
        :type solution: dict
        :return: processed solution and the time it took to process it
        :rtype: tuple(list, bool)
        """
        start_time = time() * 1000
        nodes = list(self.application.nodes())
        start = ((0, 0), 1, 1)
        # fill route with None values
        route = [None] * int((len(self.application) - 1) / 2 + 1)
        visited_seams = []
        # get nodes from sample
        if sum(value == 1 for value in solution.values()) > len(route):
            logging.warning("Result is longer than route! This might be problematic!")
        # NOTE: Prevent duplicate node entries by enforcing only one occurrence per node along route
        for (node, config, tool, timestep), val in solution.items():
            if val and (node[0] not in visited_seams):
                if route[timestep] is not None:
                    visited_seams.remove(route[timestep][0][0])
                route[timestep] = (node, config, tool)
                visited_seams.append(node[0])

        # run heuristic replacing None values
        if None in route:
            # get not assigned nodes
            logging.info(f"Route until now is: {route}")
            nodes_unassigned = [(node, 1, 1) for node in nodes if node[0] not in visited_seams]
            nodes_unassigned = list(np.random.permutation(nodes_unassigned))
            logging.info(nodes_unassigned)
            logging.info(visited_seams)
            logging.info(nodes)
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
        parsed_route = ' ->\n'.join(
            [f' Node {visit[0][1]} of Seam {visit[0][0]} using config {visit[1]} & tool {visit[2]}' for visit in route])
        logging.info(f"Route found:\n{parsed_route}")
        return route, round(time() * 1000 - start_time, 3)

    def validate(self, solution: list) -> (bool, float):
        """
        Checks if all seams and the home position are visited for a given solution.

        :param solution: list containing the nodes of the solution
        :type solution: list
        :return: Boolean whether the solution is valid, time it took to validate
        :rtype: tuple(bool, float)
        """
        # Check if all seams are visited in route
        start = time() * 1000
        visited_seams = list(set(list([seam[0][0] for seam in solution if seam is not None])))  # pylint: disable=R1728

        if len(visited_seams) == len(solution):
            logging.info(
                f"All {len(solution) - 1} seams and the base node got visited (We only need to visit 1 node per seam)")
            return True, round(time() * 1000 - start, 3)
        else:
            logging.error(f"Only {len(visited_seams) - 1} got visited")
            return False, round(time() * 1000 - start, 3)

    def evaluate(self, solution: list) -> (float, float):
        """
        Calculates the tour length for a given valid tour.

        :param solution: list containing the nodes of the solution
        :type solution: list
        :return: Tour length, time it took to calculate the tour length
        :rtype: tuple(float, float)
        """
        start = time() * 1000
        # get the total distance
        total_dist = 0
        for idx, _ in enumerate(solution[:-1]):
            edge = next(item for item in list(self.application[solution[idx][0]][solution[idx + 1][0]].values()) if
                        item["c_start"] == solution[idx][1] and item["t_start"] == solution[idx][2] and item[
                            "c_end"] == solution[idx + 1][1] and item["t_end"] == solution[idx + 1][2])
            dist = edge['weight']
            total_dist += dist

        logging.info(f"Total distance (without return): {total_dist}")

        # add distance between start and end point to complete cycle
        return_edge = next(item for item in list(self.application[solution[0][0]][solution[-1][0]].values()) if
                           item["c_start"] == solution[0][1] and item["t_start"] == solution[0][2] and item[
                               "c_end"] == solution[-1][1] and item["t_end"] == solution[-1][2])
        return_distance = return_edge['weight']
        logging.info(f"Distance between start and end: {return_distance}")

        # get distance for full cycle
        distance = total_dist + return_distance
        logging.info(f"Total distance (including return): {distance}")

        return distance, round(time() * 1000 - start, 3)

    def save(self, path: str, iter_count: int) -> None:
        nx.write_gpickle(self.application, f"{path}/graph_iter_{iter_count}.gpickle")
