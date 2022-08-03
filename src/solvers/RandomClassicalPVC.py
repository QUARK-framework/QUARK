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
import random

import networkx

from devices.Local import Local
from solvers.Solver import *


class RandomPVC(Solver):
    """
    Classical Random Solver for the PVC problem.
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.device_options = ["Local"]

    def get_device(self, device_option: str) -> any:
        if device_option == "Local":
            return Local()
        else:
            raise NotImplementedError(f"Device Option {device_option} not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns empty dict as this solver has no configurable settings

        :return: empty dict
        :rtype: dict
        """
        return {

        }

    class Config(TypedDict):
        """
        Empty config as this solver has no configurable settings
        """
        pass

    def run(self, mapped_problem: networkx.Graph, device_wrapper: any, config: Config, **kwargs: dict) -> (dict, float):
        """
        Solve the PVC graph in a greedy fashion.

        :param mapped_problem: graph representing a PVC problem
        :type mapped_problem: networkx.Graph
        :param device_wrapper: Local device
        :type device_wrapper: any
        :param config: empty dict
        :type config: Config
        :param kwargs: no additionally settings needed
        :type kwargs: any
        :return: Solution, the time it took to compute it and optional additional information
        :rtype: tuple(list, float, dict)
        """

        # Need to to deep copy since we are modifying the graph in this function. Else the next repetition would work
        # with a different graph
        mapped_problem = mapped_problem.copy()
        start = time() * 1000
        # We always start at the base node
        current_node = ((0, 0), 1, 1)
        idx = 1

        tour = dict()
        tour[current_node + (0,)] = 1  # (0,) is the timestep we visit this node

        # Tour needs to cover all nodes, if there are 2 nodes left we can finish since these 2 nodes belong to the same seam
        while len(mapped_problem.nodes) > 2:
            # Get the random neighbor edge from the current node
            next_node = random.choice([x for x in mapped_problem.edges(current_node[0], data=True) if
                                       x[2]['c_start'] == current_node[1] and x[2]['t_start'] == current_node[2]])
            next_node = (next_node[1], next_node[2]["c_end"], next_node[2]["t_end"])

            # Make the step - add distance to cost, add best node to tour,
            tour[next_node + (idx,)] = 1

            # Remove all node of that seam
            to_remove = [x for x in mapped_problem.nodes if x[0] == current_node[0][0]]
            for node in to_remove:
                mapped_problem.remove_node(node)
            current_node = next_node
            idx += 1

        # Tour needs to look like {((0, 0), 1, 1, 0): 1, ((3, 1), 1, 0, 1): 1, ((2, 1), 1, 1, 2): 1, ((4, 4), 1, 1, 3): 1}
        # ((0, 0), 1, 1, 0): 1 = ((seam, node), config, tool, timestep): yes we visit this
        return tour, round(time() * 1000 - start, 3), {}
