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
import logging
from collections import defaultdict
from typing import TypedDict, Union
from time import time

import networkx

from applications.Mapping import *
from solvers.Annealer import Annealer


class Qubo(Mapping):
    """
    QUBO formulation for the PVC

    """
    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.solver_options = ["Annealer"]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this mapping

        :return:
                 .. code-block:: python

                     return {
                                "lagrange_factor": {
                                    "values": [0.75, 1.0, 1.25],
                                    "description": "By which factor would you like to multiply your lagrange?"
                                }
                            }

        """
        return {
            "lagrange_factor": {
                "values": [0.75, 1.0, 1.25],
                "description": "By which factor would you like to multiply your lagrange?"
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python

             lagrange_factor: float

        """
        lagrange_factor: float

    def map(self, graph: networkx.Graph, config: Config) -> (dict, float):
        """
        Maps the networkx graph to a QUBO formulation.

        :param graph: a networkx graph
        :type graph: networkx.Graph
        :param config: config with the parameters specified in Config class
        :type config: Config
        :return: dict with the QUBO, time it took to map it
        :rtype: tuple(dict, float)
        """
        start = time() * 1000
        lagrange = None
        lagrange_factor = config['lagrange_factor']
        weight = 'weight'

        # Inspired by https://dnx.readthedocs.io/en/latest/_modules/dwave_networkx/algorithms/tsp.html
        n = graph.number_of_nodes()
        # we only need this number of timesteps since we only need to visit 1 node per seam (plus we start and end at the base node)
        timesteps = int((n - 1) / 2 + 1)
        # Let`s get the number of different configs and tools
        config = [x[2]['c_start'] for x in graph.edges(data=True)]
        config = list(set(config + [x[2]['c_end'] for x in graph.edges(data=True)]))

        tool = [x[2]['t_start'] for x in graph.edges(data=True)]
        tool = list(set(tool + [x[2]['t_end'] for x in graph.edges(data=True)]))

        if lagrange is None:
            # If no lagrange parameter provided, set to 'average' tour length.
            # Usually a good estimate for a lagrange parameter is between 75-150%
            # of the objective function value, so we come up with an estimate for
            # tour length and use that.
            if graph.number_of_edges() > 0:
                weights = [x[2]['weight'] for x in graph.edges(data=True)]
                # At the moment we need to filter out the very high artificial values we added during generate_problem
                # as this would mess up the lagrange
                weights = list(filter(lambda a: a != max(weights), weights))
                lagrange = sum(weights) / len(weights) * timesteps
            else:
                lagrange = 2

        lagrange = lagrange * lagrange_factor

        logging.info(f"Selected lagrange is: {lagrange}")

        # some input checking
        if n in (1, 2) or len(graph.edges) < n * (n - 1) // 2:
            msg = "graph must be a complete graph with at least 3 nodes or empty"
            raise ValueError(msg)

        # Creating the QUBO
        q = defaultdict(float)

        # We need to implement the following constrains:
        # Only visit 1 node of each seam
        # Don`t visit nodes twice (even if their config/tool is different)
        # We only need to visit base node at the once since this path from last node to base node is unique anyway

        # Constraint to only visit a node/seam once
        for node in graph:  # for all nodes in the graph
            for pos_1 in range(timesteps):  # for number of timesteps
                for t_start in tool:
                    for c_start in config:
                        q[((node, c_start, t_start, pos_1),
                           (node, c_start, t_start,
                            pos_1))] -= lagrange  # lagrange  # nodes to itself on the same timestep
                        for t_end in tool:
                            # for all configs and tools
                            for c_end in config:
                                if c_start != c_end or t_start != t_end:
                                    q[((node, c_start, t_start, pos_1),
                                       (node, c_end, t_end, pos_1))] += 1.0 * lagrange
                                for pos_2 in range(pos_1 + 1,
                                                   timesteps):  # For each following timestep set value for u -> u
                                    q[((node, c_start, t_start, pos_1),
                                       (node, c_end, t_end,
                                        pos_2))] += 2.0 * lagrange  # penalize visiting same node again in another timestep

                                    # penalize visiting other node of same seam
                                    if node != (0, 0):
                                        # (0,0) is the base node, it is not a seam
                                        # get the other nodes of the same seam
                                        other_seam_nodes = [x for x in graph.nodes if x[0] == node[0] and x[1] != node]
                                        for other_seam_node in other_seam_nodes:
                                            q[((node, c_start, t_start, pos_1),
                                               (other_seam_node, c_end, t_end,
                                                pos_2))] += 2.0 * lagrange  # penalize visiting other node of same seam

        # Constraint to only visit a single node in a single timestep
        for pos in range(timesteps):  # for all timesteps
            for node_1 in graph:  # for all nodes
                for t_start in tool:
                    for c_start in config:
                        q[((node_1, c_start, t_start, pos),
                           (node_1, c_start, t_start, pos))] -= lagrange
                        for t_end in tool:
                            for c_end in config:
                                # if c_start != c_end or t_start != t_end:
                                #     Q[((node_1, c_start, t_start, pos),
                                #        (node_1, c_end, t_end, pos))] += lagrange
                                for node_2 in set(graph) - {node_1}:  # for all nodes except node1 -> node1
                                    # QUBO coefficient is 2*lagrange, but we are placing this value
                                    # above *and* below the diagonal, so we put half in each position.
                                    q[((node_1, c_start, t_start, pos), (node_2, c_end, t_end,
                                                                         pos))] += lagrange  # penalize from node1 -> node2 in the same timestep

        # Objective that minimizes distance
        for u, v in itertools.combinations(graph.nodes, 2):
            for pos in range(timesteps):
                for t_start in tool:
                    for t_end in tool:
                        for c_start in config:
                            for c_end in config:
                                nextpos = (pos + 1) % timesteps
                                edge_u_v = next(item for item in list(graph[u][v].values()) if
                                                item["c_start"] == c_start and item["t_start"] == t_start and item[
                                                    "c_end"] == c_end and item["t_end"] == t_end)
                                # since it is the other direction we switch start and end of tool and config
                                edge_v_u = next(item for item in list(graph[v][u].values()) if
                                                item["c_start"] == c_end and item["t_start"] == t_end and item[
                                                    "c_end"] == c_start and item["t_end"] == t_start)
                                # going from u -> v
                                q[((u, c_start, t_start, pos), (v, c_end, t_end, nextpos))] += edge_u_v[weight]

                                # going from v -> u
                                q[((v, c_end, t_end, pos), (u, c_start, t_start, nextpos))] += edge_v_u[weight]

        logging.info("Created Qubo")

        return {"Q": q}, round(time() * 1000 - start, 3)

    def get_solver(self, solver_option: str) -> Union[Annealer]:

        if solver_option == "Annealer":
            return Annealer()
        else:
            raise NotImplementedError(f"Solver Option {solver_option} not implemented")
