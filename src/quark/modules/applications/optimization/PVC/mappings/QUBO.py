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
from collections import defaultdict
from typing import TypedDict
import logging

import networkx as nx

from quark.modules.applications.Mapping import Mapping, Core
from quark.utils import start_time_measurement, end_time_measurement


class QUBO(Mapping):
    """
    QUBO formulation for the PVC.
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__()
        self.submodule_options = ["Annealer"]

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module.

        :return: List of dictionaries with requirements of this module
        """
        return [{"name": "networkx", "version": "3.4.2"}]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this mapping.

        :return: Dictionary containing parameter options
        .. code-block:: python

            return {
                    "lagrange_factor": {
                        "values": [0.75, 1.0, 1.25],
                        "description": "By which factor would you like to multiply your Lagrange?"
                    }
                }

        """
        return {
            "lagrange_factor": {
                "values": [0.75, 1.0, 1.25],
                "description": "By which factor would you like to multiply your Lagrange?"
            }
        }

    class Config(TypedDict):
        """
        Configuration attributes of QUBO mapping.

        Attributes:
             lagrange_factor (float): Factor to multiply the Langrange.

        """
        lagrange_factor: float

    def map(self, problem: nx.Graph, config: Config) -> tuple[dict, float]:
        """
        Maps the networkx graph to a QUBO formulation.

        :param problem: Networkx graph representing the PVC problem
        :param config: Config dictionary with the mapping configuration
        :return: Tuple containing the QUBO dictionary and the time it took to map it
        """
        # Inspired by https://dnx.readthedocs.io/en/latest/_modules/dwave_networkx/algorithms/tsp.html
        start = start_time_measurement()
        lagrange_factor = config['lagrange_factor']

        # Estimate lagrange if not provided
        n = problem.number_of_nodes()
        timesteps = int((n - 1) / 2 + 1)

        # Get the number of different configs and tools
        config = [x[2]['c_start'] for x in problem.edges(data=True)]
        config = list(set(config + [x[2]['c_end'] for x in problem.edges(data=True)]))

        tool = [x[2]['t_start'] for x in problem.edges(data=True)]
        tool = list(set(tool + [x[2]['t_end'] for x in problem.edges(data=True)]))

        if problem.number_of_edges() > 0:
            weights = [x[2]['weight'] for x in problem.edges(data=True)]
            weights = list(filter(lambda a: a != max(weights), weights))
            lagrange = sum(weights) / len(weights) * timesteps
        else:
            lagrange = 2

        lagrange *= lagrange_factor
        logging.info(f"Selected lagrange is: {lagrange}")

        if n in (1, 2) or len(problem.edges) < n * (n - 1) // 2:
            msg = "graph must be a complete graph with at least 3 nodes or empty"
            raise ValueError(msg)

        # Creating the QUBO
        q = defaultdict(float)

        # We need to implement the following constrains:
        # Only visit 1 node of each seam
        # Don`t visit nodes twice (even if their config/tool is different)
        # We only need to visit base node at the once since this path from last node to base node is unique anyway

        # Constraint to only visit a node/seam once
        for node in problem:  # for all nodes in the graph
            for pos_1 in range(timesteps):  # for number of timesteps
                for t_start in tool:
                    for c_start in config:
                        q[((node, c_start, t_start, pos_1), (node, c_start, t_start, pos_1))] -= lagrange
                        for t_end in tool:
                            # For all configs and tools
                            for c_end in config:
                                if c_start != c_end or t_start != t_end:
                                    q[((node, c_start, t_start, pos_1), (node, c_end, t_end, pos_1))] += 1.0 * lagrange
                                for pos_2 in range(pos_1 + 1, timesteps):
                                    # Penalize visiting same node again in another timestep
                                    q[((node, c_start, t_start, pos_1), (node, c_end, t_end, pos_2))] += 2.0 * lagrange
                                    # Penalize visiting other node of same seam
                                    if node != (0, 0):
                                        # (0,0) is the base node, it is not a seam
                                        # Get the other nodes of the same seam
                                        other_seam_nodes = [
                                            x for x in problem.nodes if x[0] == node[0] and x[1] != node
                                        ]
                                        for other_seam_node in other_seam_nodes:
                                            # Penalize visiting other node of same seam
                                            q[((node, c_start, t_start, pos_1),
                                               (other_seam_node, c_end, t_end, pos_2))] += 2.0 * lagrange

        # Constraint to only visit a single node in a single timestep
        for pos in range(timesteps):
            for node_1 in problem:
                for t_start in tool:
                    for c_start in config:
                        q[((node_1, c_start, t_start, pos), (node_1, c_start, t_start, pos))] -= lagrange
                        for t_end in tool:
                            for c_end in config:
                                for node_2 in set(problem) - {node_1}:  # for all nodes except node1 -> node1
                                    q[((node_1, c_start, t_start, pos), (node_2, c_end, t_end, pos))] += lagrange

        # Objective that minimizes distance
        for u, v in itertools.combinations(problem.nodes, 2):
            for pos in range(timesteps):
                for t_start in tool:
                    for t_end in tool:
                        for c_start in config:
                            for c_end in config:
                                nextpos = (pos + 1) % timesteps
                                edge_u_v = next(
                                    item for item in list(problem[u][v].values())
                                    if item["c_start"] == c_start and item["t_start"] == t_start and
                                    item["c_end"] == c_end and item["t_end"] == t_end
                                )
                                # Since it is the other direction we switch start and end of tool and config
                                edge_v_u = next(
                                    item for item in list(problem[v][u].values())
                                    if item["c_start"] == c_end and item["t_start"] == t_end and
                                    item["c_end"] == c_start and item["t_end"] == t_start
                                )
                                # Going from u -> v
                                q[((u, c_start, t_start, pos), (v, c_end, t_end, nextpos))] += edge_u_v['weight']
                                # Going from v -> u
                                q[((v, c_end, t_end, pos), (u, c_start, t_start, nextpos))] += edge_v_u['weight']

        logging.info("Created Qubo")

        return {"Q": q}, end_time_measurement(start)

    def get_default_submodule(self, option: str) -> Core:
        """
        Returns the default submodule based on the provided option.

        :param option: Option specifying the submodule
        :return: Instance of the corresponding submodule
        :raises NotImplementedError: If the option is not recognized
        """
        if option == "Annealer":
            from quark.modules.solvers.Annealer import Annealer  # pylint: disable=C0415
            return Annealer()
        else:
            raise NotImplementedError(f"Solver Option {option} not implemented")
