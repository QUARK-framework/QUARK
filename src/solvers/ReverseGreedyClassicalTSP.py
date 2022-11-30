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
from networkx.algorithms import approximation as approx
from devices.Local import Local
from solvers.Solver import *


class ReverseGreedyClassicalTSP(Solver):
    """
    Classical Reverse Greedy Solver for the TSP. We take the worst choice at each step.
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

    def run(self, mapped_problem: networkx.Graph, device_wrapper: any, config: Config, **kwargs: any) -> (dict, float):
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
        :return: Solution, the time it took to compute it and optional additional information
        :rtype: tuple(list, float, dict)
        """

        # Need to deep copy since we are modifying the graph in this function. Else the next repetition would work
        # with a different graph
        mapped_problem = mapped_problem.copy()
        # Let's flip the edge weights to take the worst node every time instead of the best
        for u, v, d in mapped_problem.edges(data=True):
            d['weight'] = -1.0 * d['weight']
        start = time() * 1000

        tour = approx.greedy_tsp(mapped_problem)

        # We remove the duplicate node as we don't want a cycle
        # https://stackoverflow.com/a/7961390/10456906
        tour = list(dict.fromkeys(tour))

        # Parse tour so that it can be processed later
        result = dict()
        for idx, node in enumerate(tour):
            result[(node, idx)] = 1
        # Tour needs to look like
        return result, round(time() * 1000 - start, 3), {}
