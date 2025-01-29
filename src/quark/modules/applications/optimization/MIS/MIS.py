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
import pickle
from typing import TypedDict

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from quark.modules.applications.Application import Core
from quark.modules.applications.optimization.Optimization import Optimization
from quark.modules.applications.optimization.MIS.data.graph_layouts import generate_hexagonal_graph
from quark.utils import start_time_measurement, end_time_measurement

# Define R_rydberg
R_rydberg = 9.75


class MIS(Optimization):
    """
    The maximum independent set (MIS) problem is a combinatorial optimization problem that seeks to find the largest
    subset of vertices in a graph such that no two vertices are adjacent. MIS has numerous application in computer
    science, network design, resource allocation, and even in physics, where finding optimal configurations can
    solve fundamental problems related to stability and energy minimization.

    In a graph, the maximum independent set represents a set of nodes such that no two nodes share an edge. This
    property makes it a key element in various optimization scenarios. Due to the problem's combinatorial nature,
    it becomes computationally challenging, especially for large graphs, often requiring heuristic or approximate
    solutions.

    In the context of QUARK, we employ quantum-inspired approaches and state-of-the-art classical algorithms to
    tackle the problem. The graph is generated based on user-defined parameters such as size, spacing, and
    filling fraction, which affect the complexity and properties of the generated instance.
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__("MIS")
        self.submodule_options = ["QIRO", "NeutralAtom"]
        # TODO add more solvers like classical heuristics, VQE, QAOA, etc.
        self.depending_parameters = True
        self.graph = None

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module.

        :return: List of dict with requirements of this module
        """
        return []

    def get_solution_quality_unit(self) -> str:
        """
        Returns the unit of measurement for solution quality.

        :return: The unit of measure for solution quality
        """
        return "Set size"

    def get_default_submodule(self, option: str) -> Core:
        """
        Returns the default submodule based on the provided option.

        :param option: Option specifying the submodule
        :return: Instance of the corresponding submodule
        :raises NotImplementedError: If the option is not recognized
        """
        if option == "QIRO":
            from quark.modules.applications.optimization.MIS.mappings.QIRO import QIRO  # pylint: disable=C0415
            return QIRO()
        elif option == "NeutralAtom":
            from quark.modules.applications.optimization.MIS.mappings.NeutralAtom import NeutralAtom  # pylint: disable=C0415
            return NeutralAtom()
        else:
            raise NotImplementedError(f"Mapping Option {option} not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this application.

        :return: Configuration dictionary for this application
        .. code-block:: python

        return {
                "size": {
                    "values": [1, 5, 10, 15],
                    "custom_input": True,
                    "allow_ranges": True,
                    "postproc": int,
                    "description": "How large should your graph be?"
                },
                "graph_type": {
                    "values": ["hexagonal", "erdosRenyi"],
                    "postproc": str,
                    "description": "Do you want a hexagonal or an Erdos-Renyi graph?",
                    "depending_submodule": True
                }
            }
        """
        return {
            "size": {
                "values": [1, 5, 10, 15],
                "custom_input": True,
                "allow_ranges": True,
                "postproc": int,
                "description": "How large should your graph be?"
            },
            "graph_type": {
                "values": ["hexagonal", "erdosRenyi"],
                "postproc": str,
                "description": "Do you want a hexagonal or an Erdos-Renyi graph?",
                "depending_submodule": True
            }
        }

    def get_available_submodules(self, option: list) -> list:
        """
        Changes mapping options based on selection of graphs.

        :param option: List of chosen graph type
        :return: List of available submodules
        """
        if option == ["hexagonal"]:
            return ["QIRO", "NeutralAtom"]
        else:
            return ["QIRO"]

    def get_depending_parameters(self, option: str, config: dict) -> dict:
        """
        Returns parameters necessary for chosen problem option.

        :param option: The chosen option
        :param config: The current config
        :return: The parameters for the given option
        """

        more_params = {
            "filling_fraction": {
                "values": [x / 10 for x in range(2, 11, 2)],
                "custom_input": True,
                "allow_ranges": True,
                "postproc": float,
                "description": "What should be the filling fraction of the hexagonal graph / p of erdosRenyi graph?"
            }}
        if option == "QIRO":
            more_params["seed"] = {
                "values": ["No"],
                "custom_input": True,
                "description": "Do you want to set a seed? If yes, please set an integer number"
            }

        elif option == "NeutralAtom":
            pass  # No additional parameters needed at the moment
        else:
            raise NotImplementedError(f"Option {option} not implemented")
        if "hexagonal" in config["graph_type"]:
            more_params["spacing"] = {
                "values": [x / 10 for x in range(3, 11, 2)],
                "custom_input": True,
                "allow_ranges": True,
                "postproc": float,
                "description": "How much space do you want between your nodes, relative to Rydberg distance?"
            }
        param_to_return = {}
        for key, value in more_params.items():
            if key not in config:
                param_to_return[key] = value

        return param_to_return

    class Config(TypedDict):
        """
        Configuration attributes for generating an MIS problem.

        Attributes:
            size (int): The number of nodes in the graph.
            spacing (float): The spacing between nodes in the graph.
            filling_fraction (float): The fraction of available places in the lattice filled with nodes
        """
        size: int
        spacing: float
        filling_fraction: float

    def generate_problem(self, config: Config) -> nx.Graph:
        """
        Generates a graph to solve the MIS problem for.

        :param config: Config specifying the size and connectivity for the problem
        :return: Networkx graph representing the problem
        """
        if config is None:
            logging.warning("No config provided, using default values: graph_type='hexagonal', size=3, spacing=1,"
                            "filling_fraction=0.5")
            config = {"graph_type": "hexagonal", "size": 3, "spacing": 1, "filling_fraction": 0.5}

        graph_type = config.get('graph_type')
        size = config.get('size')
        filling_fraction = config.get('filling_fraction')

        if graph_type == "erdosRenyi":
            gseed = config.get("seed")

            if gseed == "No":
                graph = nx.erdos_renyi_graph(size, filling_fraction)

            else:
                try:
                    gseed = int(gseed)
                except ValueError:
                    logging.warning(f"Please select an integer number as seed for the Erdos-Renyi graph instead of "
                                    f"'{gseed}'. The seed is instead set to 0.")
                    gseed = 0
                graph = nx.erdos_renyi_graph(size, filling_fraction, seed=gseed)
            logging.info("Created MIS problem with the nx.erdos_renyi graph method, with the following attributes:")
            logging.info(f" - Graph size: {size}")
            logging.info(f" - p: {filling_fraction}")
            logging.info(f" - seed: {gseed}")

        else:
            if config.get('spacing') is None:
                spacing = 0.5
            else:
                spacing = config.get('spacing')
            graph = generate_hexagonal_graph(n_nodes=size,
                                             spacing=spacing * R_rydberg,
                                             filling_fraction=filling_fraction)
            logging.info("Created MIS problem with the generate hexagonal graph method, with the following attributes:")
            logging.info(f" - Graph size: {size}")
            logging.info(f" - Spacing: {spacing * R_rydberg}")
            logging.info(f" - Filling fraction: {filling_fraction}")

        self.graph = graph
        return graph.copy()

    def validate(self, solution: list) -> tuple[bool, float]:
        """
        Checks if the solution is an independent set.

        :param solution: List containing the nodes of the solution
        :return: Boolean whether the solution is valid and time it took to validate
        """
        start = start_time_measurement()
        is_valid = True

        nodes = list(self.graph.nodes())
        edges = list(self.graph.edges())

        # Check if the solution is independent
        is_independent = all((u, v) not in edges for u, v in edges if u in solution and v in solution)
        if is_independent:
            logging.info("The solution is independent.")
        else:
            logging.warning("The solution is not independent.")
            is_valid = False

        # Check if the solution is a set
        solution_set = set(solution)
        is_set = len(solution_set) == len(solution)
        if is_set:
            logging.info("The solution is a set.")
        else:
            logging.warning("The solution is not a set.")
            is_valid = False

        # Check if the solution is a subset of the original nodes
        is_subset = all(node in nodes for node in solution)
        if is_subset:
            logging.info("The solution is a subset of the problem.")
        else:
            logging.warning("The solution is not a subset of the problem.")
            is_valid = False

        return is_valid, end_time_measurement(start)

    def evaluate(self, solution: list) -> tuple[int, float]:
        """
        Calculates the size of the solution.

        :param solution: List containing the nodes of the solution
        :return: Set size, time it took to calculate the set size
        """
        start = start_time_measurement()
        set_size = len(solution)

        logging.info(f"Size of solution: {set_size}")

        return set_size, end_time_measurement(start)

    def save(self, path: str, iter_count: int) -> None:
        """
        Saves the generated problem graph to a file.

        :param path: Path to save the problem graph
        :param iter_count: Iteration count for file versioning
        """
        with open(f"{path}/graph_iter_{iter_count}.gpickle", "wb") as file:
            pickle.dump(self.graph, file, pickle.HIGHEST_PROTOCOL)

    def visualize_solution(self, processed_solution: list[int], path: str):
        """
        Plot the problem graph with the solution nodes highlighted

        :param processed_solution: The solution already processed by :func:`process_solution`, a list of visited node IDs in order of being visited.
        :param path: File path for the plot
        :returns: None
        """
        NODE_SIZE = 300   # Default=300
        EDGE_WIDTH = 1.0  # Default=1.0
        FONT_SIZE = 12    # Default=12
        COLOR_INCLUDED = "red"
        COLOR_EXCLUDED = "gray"

        G = self.graph
        included_nodes = [node for node in G.nodes() if node in processed_solution]
        excluded_nodes = [node for node in G.nodes() if node not in processed_solution]
        pos = nx.circular_layout(G)
        included_pos = {n: n for n, _ in pos.items() if n in processed_solution}
        excluded_pos = {n: n for n, _ in pos.items() if n not in processed_solution}
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker='o',
                ls="None",
                label="Included",
                markerfacecolor=COLOR_INCLUDED,
                markeredgewidth=0,
                markersize=10),
            Line2D(
                [0],
                [0],
                marker='o',
                ls="None",
                label="Excluded",
                markerfacecolor=COLOR_EXCLUDED,
                markeredgewidth=0,
                markersize=10)
        ]

        nx.draw_networkx_nodes(G, pos, nodelist=included_nodes, node_size=NODE_SIZE, node_color=COLOR_INCLUDED)
        nx.draw_networkx_nodes(G, pos, nodelist=excluded_nodes, node_size=NODE_SIZE, node_color=COLOR_EXCLUDED)
        nx.draw_networkx_labels(G, pos, included_pos, font_size=FONT_SIZE, font_weight="bold")
        nx.draw_networkx_labels(G, pos, excluded_pos, font_size=FONT_SIZE)
        nx.draw_networkx_edges(G, pos, width=EDGE_WIDTH)

        plt.legend(handles=legend_elements)
        plt.savefig(path)
        plt.close()
