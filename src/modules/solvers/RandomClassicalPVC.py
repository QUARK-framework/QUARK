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
import networkx as nx

from modules.solvers.Solver import Solver
from modules.Core import Core
from utils import start_time_measurement, end_time_measurement


class RandomPVC(Solver):
    """
    Classical Random Solver for the PVC problem.
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__()
        self.submodule_options = ["Local"]

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module.

        :return: List of dict with requirements of this module
        """
        return [{"name": "networkx", "version": "3.2.1"}]

    def get_default_submodule(self, option: str) -> Core:
        """
        Returns the default submodule based on the provided option.

        :param option: The name of the submodule
        :return: Instance of the default submodule
        """
        if option == "Local":
            from modules.devices.Local import Local  # pylint: disable=C0415
            return Local()
        else:
            raise NotImplementedError(f"Device Option {option} not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns empty dictionary as this solver has no configurable settings.

        :return: Empty dict
        """
        return {}

    class Config(TypedDict):
        """
        Empty config as this solver has no configurable settings.
        """
        pass

    def run(self, mapped_problem: nx.Graph, device_wrapper: any, config: Config, **kwargs: dict) \
            -> tuple[dict, float, dict]:
        """
        Solve the PVC graph in a greedy fashion.

        :param mapped_problem: Graph representing a PVC problem
        :param device_wrapper: Local device
        :param config: Empty dict
        :param kwargs: No additionally settings needed
        :return: Solution, the time it took to compute it and optional additional information
        """
        # Deep copy since we are modifying the graph. This ensures that the original graph remains unchanged
        # with a different graph
        mapped_problem = mapped_problem.copy()
        start = start_time_measurement()

        # We always start at the base node
        current_node = ((0, 0), 1, 1)
        idx = 1

        tour = {current_node + (0,): 1}  # (0,) is the timestep we visit this node

        # Tour needs to cover all nodes, if there are 2 nodes left we can finish since these 2 nodes belong
        # to the same seam
        while len(mapped_problem.nodes) > 2:
            # Get the random neighbor edge from the current node
            next_node = random.choice([
                x for x in mapped_problem.edges(current_node[0], data=True)
                if x[1][0] != current_node[0][0] and x[2]['c_start'] == current_node[1] and x[2]['t_start']
                == current_node[2]
            ])
            next_node = (next_node[1], next_node[2]["c_end"], next_node[2]["t_end"])

            # Make the step - add distance to cost, add the best node to tour,
            tour[next_node + (idx,)] = 1

            # Remove all node of that seam
            to_remove = [x for x in mapped_problem.nodes if x[0] == current_node[0][0]]
            for node in to_remove:
                mapped_problem.remove_node(node)

            current_node = next_node
            idx += 1

        # Tour needs to look like {((0, 0), 1, 1, 0): 1,((3, 1), 1, 0, 1): 1,((2, 1), 1, 1, 2): 1,((4, 4), 1, 1, 3): 1}
        # ((0, 0), 1, 1, 0): 1 = ((seam, node), config, tool, timestep): yes we visit this
        return tour, end_time_measurement(start), {}
