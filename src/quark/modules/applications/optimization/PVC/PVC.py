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
from typing import TypedDict
import pickle
import logging
import os

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from quark.modules.applications.Application import Core
from quark.modules.applications.optimization.Optimization import Optimization
from quark.utils import start_time_measurement, end_time_measurement


class PVC(Optimization):
    """
    In modern vehicle manufacturing, robots take on a significant workload, including performing welding
    jobs, sealing welding joints, or applying paint to the car body. While the robot’s tasks vary widely,
    the objective remains the same: Perform a job with the highest possible quality in the shortest amount
    of time, optimizing efficiency and productivity on the manufacturing line.

    For instance, to protect a car’s underbody from corrosion, exposed welding seams are sealed
    by applying a polyvinyl chloride layer (PVC). The welding seams need to be traversed by a robot to
    apply the material. It is related to TSP, but different and even more complex in some aspects.

    The problem of determining the optimal route for robots to traverse all seams shares similarities
    with Traveling Salesman Problem (TSP), as it involves finding the shortest possible route to
    visit multiple locations. However, it introduces additional complexities, such as different tool
    and configuration requirements for each seam, making it an even more challenging problem to solve.
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__("PVC")
        self.submodule_options = [
            "Ising", "QUBO", "GreedyClassicalPVC", "ReverseGreedyClassicalPVC", "RandomPVC"
        ]

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module.

        :return: List of dict with requirements of this module
        """
        return [
            {"name": "networkx", "version": "3.4.2"},
            {"name": "numpy", "version": "1.26.4"}
        ]

    def get_solution_quality_unit(self) -> str:
        """
        Returns the unit of measure for solution quality.

        :return: Unit of measure for solution quality
        """
        return "Tour cost"

    def get_default_submodule(self, option: str) -> Core:
        """
        Returns the default submodule based on the provided option.

        :param option: Option specifying the submodule
        :return: Instance of the corresponding submodule
        :raises NotImplementedError: If the option is not recognized
        """
        if option == "Ising":
            from quark.modules.applications.optimization.PVC.mappings.ISING import Ising  # pylint: disable=C0415
            return Ising()
        elif option == "QUBO":
            from quark.modules.applications.optimization.PVC.mappings.QUBO import QUBO  # pylint: disable=C0415
            return QUBO()
        elif option == "GreedyClassicalPVC":
            from quark.modules.solvers.GreedyClassicalPVC import GreedyClassicalPVC  # pylint: disable=C0415
            return GreedyClassicalPVC()
        elif option == "ReverseGreedyClassicalPVC":
            from quark.modules.solvers.ReverseGreedyClassicalPVC import ReverseGreedyClassicalPVC  # pylint: disable=C0415
            return ReverseGreedyClassicalPVC()
        elif option == "RandomPVC":
            from quark.modules.solvers.RandomClassicalPVC import RandomPVC  # pylint: disable=C0415
            return RandomPVC()
        else:
            raise NotImplementedError(f"Mapping Option {option} not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this application.

        :return: Dictionary containing parameter options
        .. code-block:: python

            return {
                    "seams": {
                        "values": list(range(1, 18)),
                        "description": "How many seams should be sealed?"
                    }
                }
        """
        return {
            "seams": {
                "values": list(range(1, 18)),
                # In the current implementation the graph can only be as large as the reference input graph
                "description": "How many seams should be sealed?"
            }
        }

    class Config(TypedDict):
        """
       Configuration attributes for PVC problem generation.

       Attributes:
        seams (int): Number of seams for the graph
        """
        seams: int

    def generate_problem(self, config: Config) -> nx.Graph:
        """
        Uses the reference graph to generate a problem for a given config.

        :param config: Config specifying the number of seams for the problem
        :return: Networkx graph representing the problem
        """
        if config is None:
            config = {"seams": 3}
        seams = config['seams']

        # Read in the original graph
        with open(os.path.join(os.path.dirname(__file__), "data", "reference_graph.gpickle"), "rb") as file:
            graph = pickle.load(file)

        # Get number of seam in graph
        seams_in_graph = list({x[0] for x in graph.nodes})
        seams_in_graph.sort()
        seams_in_graph.remove(0)  # Always need the base node 0 (which is not a seam)

        if len(seams_in_graph) < seams:
            logging.info("Too many seams! The original graph has less seams than that!")

        unwanted_seams = seams_in_graph[-len(seams_in_graph) + seams:]
        unwanted_nodes = [x for x in graph.nodes if x[0] in unwanted_seams]

        for node in unwanted_nodes:
            graph.remove_node(node)

        if not nx.is_strongly_connected(graph):
            logging.error("Graph is not connected!")
            raise ValueError("Graph is not connected!")

        # Gather unique configurations and tools
        config = [x[2]['c_start'] for x in graph.edges(data=True)]
        config = list(set(config + [x[2]['c_end'] for x in graph.edges(data=True)]))
        tool = [x[2]['t_start'] for x in graph.edges(data=True)]
        tool = list(set(tool + [x[2]['t_end'] for x in graph.edges(data=True)]))

        # Fill the rest of the missing edges with high values
        current_edges = [
            (edge[0], edge[1], edge[2]['t_start'], edge[2]['t_end'], edge[2]['c_start'], edge[2]['c_end'])
            for edge in graph.edges(data=True)
        ]
        all_possible_edges = list(itertools.product(list(graph.nodes), repeat=2))
        all_possible_edges = [
            (edges[0], edges[1], t_start, t_end, c_start, c_end)
            for edges in all_possible_edges
            for c_end in config
            for c_start in config
            for t_end in tool
            for t_start in tool if edges[0] != edges[1]
        ]

        missing_edges = [item for item in all_possible_edges if item not in current_edges]

        # Add these edges with very high values
        for edge in missing_edges:
            graph.add_edge(
                edge[0], edge[1], c_start=edge[4], t_start=edge[2], c_end=edge[5], t_end=edge[3], weight=100000
            )

        logging.info("Created PVC problem with the following attributes:")
        logging.info(f" - Number of seams: {seams}")
        logging.info(f" - Number of different configs: {len(config)}")
        logging.info(f" - Number of different tools: {len(tool)}")

        self.application = graph
        return graph.copy()

    def process_solution(self, solution: dict) -> tuple[list, float]:
        """
        Converts solution dictionary to list of visited seams.

        :param solution: Unprocessed solution
        :return: Processed solution and the time it took to process it
        """
        start_time = start_time_measurement()
        nodes = list(self.application.nodes())
        start = ((0, 0), 1, 1)
        route: list = [None] * int((len(self.application) - 1) / 2 + 1)
        visited_seams = []

        if sum(value == 1 for value in solution.values()) > len(route):
            logging.warning("Result is longer than route! This might be problematic!")

        # Prevent duplicate node entries by enforcing only one occurrence per node along route
        for (node, config, tool, timestep), val in solution.items():
            if val and (node[0] not in visited_seams):
                if route[timestep] is not None:
                    visited_seams.remove(route[timestep][0][0])
                route[timestep] = (node, config, tool)
                visited_seams.append(node[0])

        # Fill missing values in the route
        if None in route:
            logging.info(f"Route until now is: {route}")
            nodes_unassigned = [(node, 1, 1) for node in nodes if node[0] not in visited_seams]
            nodes_unassigned = list(np.random.permutation(nodes_unassigned, dtype=object))
            logging.info(nodes_unassigned)
            logging.info(visited_seams)
            logging.info(nodes)
            for idx, node in enumerate(route):
                if node is None:
                    route[idx] = nodes_unassigned.pop(0)

        # Cycle solution to start at provided start location
        if start is not None and route[0] != start:
            idx = route.index(start)
            route = route[idx:] + route[:idx]

        parsed_route = ' ->\n'.join(
            [
                f' Node {visit[0][1]} of Seam {visit[0][0]} using config '
                f' {visit[1]} & tool {visit[2]}'
                for visit in route
            ]
        )
        logging.info(f"Route found:\n{parsed_route}")

        return route, end_time_measurement(start_time)

    def validate(self, solution: list) -> tuple[bool, float]:
        """
        Checks if all seams and the home position are visited for a given solution.

        :param solution: List containing the nodes of the solution
        :return: Boolean whether the solution is valid and time it took to validate
        """
        # Check if all seams are visited in route
        start = start_time_measurement()
        visited_seams = {seam[0][0] for seam in solution if seam is not None}

        if len(visited_seams) == len(solution):
            logging.info(f"All {len(solution) - 1} seams and "
                         "the base node got visited (We only need to visit 1 node per seam)")
            return True, end_time_measurement(start)
        else:
            logging.error(f"Only {len(visited_seams) - 1} got visited")
            return False, end_time_measurement(start)

    def evaluate(self, solution: list) -> tuple[float, float]:
        """
        Calculates the tour length for a given valid tour.

        :param solution: List containing the nodes of the solution
        :return: Tour length, time it took to calculate the tour length
        """
        start = start_time_measurement()

        # Get the total distance
        total_dist = 0
        for idx, _ in enumerate(solution[:-1]):
            edge = next(
                item for item in list(self.application[solution[idx][0]][solution[idx + 1][0]].values())
                if item["c_start"] == solution[idx][1] and item["t_start"] == solution[idx][2] and
                item["c_end"] == solution[idx + 1][1] and item["t_end"] == solution[idx + 1][2]
            )
            dist = edge['weight']
            total_dist += dist
        logging.info(f"Total distance (without return): {total_dist}")

        # Add distance between start and end point to complete cycle
        return_edge = next(
            item for item in list(self.application[solution[0][0]][solution[-1][0]].values())
            if item["c_start"] == solution[0][1] and item["t_start"] == solution[0][2] and
            item["c_end"] == solution[-1][1] and item["t_end"] == solution[-1][2]
        )
        return_distance = return_edge['weight']
        logging.info(f"Distance between start and end: {return_distance}")

        # Get distance for full cycle
        distance = total_dist + return_distance
        logging.info(f"Total distance (including return): {distance}")

        return distance, end_time_measurement(start)

    def save(self, path: str, iter_count: int) -> None:
        """
        Saves the generated problem graph to a file.

        :param path: Path to save the problem graph
        :param iter_count: Iteration count for file versioning
        """
        with open(f"{path}/graph_iter_{iter_count}.gpickle", "wb") as file:
            pickle.dump(self.application, file, pickle.HIGHEST_PROTOCOL)

    def visualize_solution(self, processed_solution, path: str):
        """
        Plot a graph representing the possible locations where seams can start or end, with arrows representing either idle movements or the sealing of a seam

        :param processed_solution: The solution already processed by :func:`process_solution`, a list of tuples representing seam start points and the config and tool needed to seal the seam.
        :param path: File path for the plot
        :returns: None
        """
        NODE_SIZE = 300   # Default=300
        EDGE_WIDTH = 1.0  # Default=1.0
        FONT_SIZE = 12    # Default=12

        highest_node_id = max(node[1] for node in self.application.nodes())
        G = nx.MultiDiGraph()
        G.add_nodes_from(range(highest_node_id + 1))
        pos = nx.circular_layout(G)

        tools = set()
        configs = set()
        current_node = 0
        for ((seam1, node1), config, tool) in processed_solution[1:]:
            config = config - 1
            tools.add(tool)
            configs.add(config)
            (seam2, node2) = next((seam, node)
                                  for (seam, node) in self.application.nodes() if seam == seam1 and not node == node1)
            assert seam1 == seam2, "This is bad"
            if not current_node == node1:
                G.add_edge(current_node, node1, color=7, width=EDGE_WIDTH, style=-1)
            G.add_edge(node1, node2, color=tool, width=2 * EDGE_WIDTH, style=config)
            current_node = node2

        # The 8 here controls how many edges between the same two nodes are at
        # most drawn with spacing between them before drawing them on top of each
        # other to avoid cluttering
        connectionstyle = [f"arc3,rad={r}" for r in itertools.accumulate([0.15] * 8)]
        style_options = ["solid", "dotted", "dashed", "dashdot"]
        cmap = plt.cm.Dark2
        tools = list(tools)
        configs = list(configs)
        legend_elements = [Line2D([0],
                                  [0],
                                  color=cmap(7),
                                  lw=EDGE_WIDTH,
                                  ls=':',
                                  label="Idle Movement")] + [Patch(facecolor=cmap(i),
                                                                   label=f"Tool {i}") for i in tools] + [Line2D([0],
                                                                                                                [0],
                                                                                                                color="black",
                                                                                                                lw=2 * EDGE_WIDTH,
                                                                                                                ls=style_options[i % len(
                                                                                                                    style_options)],
                                                                                                                label=f"Config {i + 1}") for i in configs]
        colors = nx.get_edge_attributes(G, 'color').values()
        widths = nx.get_edge_attributes(G, 'width').values()
        styles = [':' if i == -1 else style_options[i % len(style_options)]
                  for i in nx.get_edge_attributes(G, 'style').values()]

        nx.draw_networkx(
            G,
            pos,
            node_size=NODE_SIZE,
            font_size=FONT_SIZE,
            style=list(styles),
            edge_color=colors,
            edge_cmap=cmap,
            width=list(widths),
            connectionstyle=connectionstyle)

        plt.legend(handles=legend_elements)
        plt.savefig(path)
        plt.close()
