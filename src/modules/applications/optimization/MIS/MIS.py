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
import pickle

import networkx

from modules.applications.Application import *
from modules.applications.optimization.Optimization import Optimization
from modules.applications.optimization.MIS.data.graph_layouts import generate_hexagonal_graph
from utils import start_time_measurement, end_time_measurement

# define R_rydberg
R_rydberg = 9.75


class MIS(Optimization):
    """
    In planning problems, there will be tasks to be done, and some of them may be mutually exclusive.
    We can translate this into a graph where the nodes are the tasks and the edges are the mutual exclusions.
    The maximum independent set (MIS) problem is a combinatorial optimization problem that seeks to find the largest
    subset of vertices in a graph such that no two vertices are adjacent.
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__("MIS")
        self.submodule_options = ["QIRO", "NeutralAtom"]
        self.depending_parameters = True

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module

        :return: list of dict with requirements of this module
        :rtype: list[dict]
        """
        return [
        ]

    def get_solution_quality_unit(self) -> str:
        return "Set size"

    def get_default_submodule(self, option: str) -> Core:
        if option == "QIRO":
            from modules.applications.optimization.MIS.mappings.QIRO import QIRO # pylint: disable=C0415
            return QIRO()
        elif option == "NeutralAtom":
            from modules.applications.optimization.MIS.mappings.NeutralAtom import NeutralAtom  # pylint: disable=C0415
            return NeutralAtom()
        else:
            raise NotImplementedError(f"Mapping Option {option} not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this application

        :return:
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

    def get_available_submodules(self, options: list) -> list:
        """
        Changes mapping options  based on selection of graphs.

        :param options: List of chosen graph type
        :type options: list
        :return: List of available submodules
        :rtype: list
        """
        if options == ["hexagonal"]:
            return ["QIRO", "NeutralAtom"]
        else:
            return ["QIRO"]

    def get_depending_parameters(self, option: str, config: dict) -> dict:
        """
        Returns parameters necessary for chosen problem option.

        :param option: The chosen option
        :type option: str
        :param config: The current config
        :type config: dict
        :return: The parameters for the given option
        :rtype: dict
        """

        more_params = {
            "filling_fraction": {
                "values": [x/10 for x in range(2, 11, 2)],
                    "custom_input": True,
                    "allow_ranges": True,
                    "postproc": float,
                    "description": "What should be the filling fraction of the hexagonal graph / p of erdosRenyi graph?"
            }}
        if option == "QIRO":
            more_params["seed"] = {
                    "values": ["No"],
                    "custom_input": True,
                    "postproc": int,
                    "description": "Do you want to set a seed? If yes, please set an integer number"
                }
            
        elif option == "NeutralAtom":
            pass  # No additional parameters needed at the moment
        else:
            raise NotImplementedError(f"Option {option} not implemented")
        if "hexagonal" in config["graph_type"]:
            more_params["spacing"] =  {
                "values": [x / 10 for x in range(3, 11, 2)],
                "custom_input": True,
                "allow_ranges": True,
                "postproc": float,
                "description": "How much space do you want between your nodes,"
                               " relative to Rydberg distance?  "
            }
        param_to_return = {}
        for key in more_params:
            if key not in config:
                param_to_return[key] = more_params[key]
        return param_to_return

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python

            size: int
            spacing: float
            filling_fraction: float

        """
        size: int
        spacing: float
        filling_fraction: float

    def generate_problem(self, config: Config) -> networkx.Graph:
        """
        Generates a graph to solve the MIS for.

        :param config: Config specifying the size and connectivity for the problem
        :type config: Config
        :return: networkx graph representing the problem
        :rtype: networkx.Graph
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
                graph = networkx.erdos_renyi_graph(size, filling_fraction)

            else:
                graph = networkx.erdos_renyi_graph(size, filling_fraction, seed=gseed)
            logging.info("Created MIS problem with the nx.erdos_renyi graph method, with the following attributes:")
            logging.info(f" - Graph size: {size}")
            logging.info(f" - p: {filling_fraction}")
            logging.info(f" - seed: {gseed}")

        else:
            if config.get('spacing') is None:
                spacing = 0.5
            else:
                spacing = config.get('spacing')
            graph = generate_hexagonal_graph(n_nodes=size, spacing=spacing * R_rydberg, filling_fraction=
                                             filling_fraction)
            logging.info("Created MIS problem with the generate hexagonal graph method, with the following attributes:")
            logging.info(f" - Graph size: {size}")
            logging.info(f" - Spacing: {spacing * R_rydberg}")
            logging.info(f" - Filling fraction: {filling_fraction}")

        self.graph = graph
        return graph.copy()

    def process_solution(self, solution: list) -> (list, float):
        """
        Returns list of visited nodes and the time it took to process the solution

        :param solution: Unprocessed solution
        :type solution: list
        :return: Processed solution and the time it took to process it
        :rtype: tuple(list, float)
        """
        start_time = start_time_measurement()

        return solution, end_time_measurement(start_time)

    def validate(self, solution: list) -> (bool, float):
        """
        Checks if the solution is an independent set

        :param solution: List containing the nodes of the solution
        :type solution: list
        :return: Boolean whether the solution is valid and time it took to validate
        :rtype: tuple(bool, float)
        """
        start = start_time_measurement()
        is_valid = True

        nodes = list(self.graph.nodes())
        edges = list(self.graph.edges())

        # TODO: Check if the solution is maximal?

        # Check if the solution is independent
        is_independent = all((u, v) not in edges for u, v in edges if u in solution and v in solution)
        if is_independent:
            logging.info("The solution is independent")
        else:
            logging.warning("The solution is not independent")
            is_valid = False

        # Check if the solution is a set
        solution_set = set(solution)
        is_set = len(solution_set) == len(solution)
        if is_set:
            logging.info("The solution is a set")
        else:
            logging.warning("The solution is not a set")
            is_valid = False

        # Check if the solution is a subset of the original nodes
        is_set = all(node in nodes for node in solution)
        if is_set:
            logging.info("The solution is a subset of the problem")
        else:
            logging.warning("The solution is not a subset of the problem")
            is_valid = False

        return is_valid, end_time_measurement(start)

    def evaluate(self, solution: list) -> (int, float):
        """
        Calculates the size of the solution

        :param solution: List containing the nodes of the solution
        :type solution: list
        :return: Set size, time it took to calculate the set size
        :rtype: tuple(int, float)
        """
        start = start_time_measurement()
        set_size = len(solution)

        logging.info(f"Size of solution: {set_size}")

        return set_size, end_time_measurement(start)

    def save(self, path: str, iter_count: int) -> None:
        with open(f"{path}/graph_iter_{iter_count}.gpickle", "wb") as file:
            pickle.dump(self.graph, file, pickle.HIGHEST_PROTOCOL)
