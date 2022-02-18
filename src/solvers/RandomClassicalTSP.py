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

import networkx
import networkx as nx
import random
from devices.Local import Local
from solvers.Solver import *


class RandomTSP(Solver):
    """
    Classical Random Solver the TSP
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
        Solve the TSP graph in a greedy fashion.

        :param mapped_problem: graph representing a TSP
        :type mapped_problem: networkx.Graph
        :param device_wrapper: Local device
        :type device_wrapper: any
        :param config: empty dict
        :type config: Config
        :param kwargs: no additionally settings needed
        :type kwargs: any
        :return: Solution and the time it took to compute it
        :rtype: tuple(dict, float)
        """

        start = time() * 1000
        source = nx.utils.arbitrary_element(mapped_problem)

        nodeset = set(mapped_problem)
        nodeset.remove(source)
        tour = [source]
        while nodeset:
            next_node = random.choice(list(nodeset))
            tour.append(next_node)
            nodeset.remove(next_node)
        tour.append(tour[0])

        # We remove the duplicate node as we don't want a cycle
        # https://stackoverflow.com/a/7961390/10456906
        tour = list(dict.fromkeys(tour))

        # Parse tour so that it can be processed later
        result = dict()
        for idx, node in enumerate(tour):
            result[(node, idx)] = 1
        # Tour needs to look like
        return result, round(time() * 1000 - start, 3)
