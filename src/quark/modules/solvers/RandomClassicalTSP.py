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

from quark.modules.solvers.Solver import Solver
from quark.modules.Core import Core
from quark.utils import start_time_measurement, end_time_measurement


class RandomTSP(Solver):
    """
    Classical Random Solver the TSP.
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
        return [{"name": "networkx", "version": "3.4.2"}]

    def get_default_submodule(self, option: str) -> Core:
        if option == "Local":
            from quark.modules.devices.Local import Local  # pylint: disable=C0415
            return Local()
        else:
            raise NotImplementedError(f"Device Option {option} not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns empty dict as this solver has no configurable settings.

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
        Solve the TSP graph in a greedy fashion.

        :param mapped_problem: Graph representing a TSP
        :param device_wrapper: Local device
        :param config: Empty dict
        :param kwargs: No additionally settings needed
        :return: Solution, the time it took to compute it and optional additional information
        """
        start = start_time_measurement()
        source = nx.utils.arbitrary_element(mapped_problem)

        nodeset = set(mapped_problem)
        nodeset.remove(source)
        tour = [source]

        while nodeset:
            next_node = random.choice(list(nodeset))
            tour.append(next_node)
            nodeset.remove(next_node)
        tour.append(tour[0])

        # Remove the duplicate node as we don't want a cycle
        tour = list(dict.fromkeys(tour))

        # Parse tour so that it can be processed later
        result = {(node, idx): 1 for idx, node in enumerate(tour)}

        return result, end_time_measurement(start), {}
