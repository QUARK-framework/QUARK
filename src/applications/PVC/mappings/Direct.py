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

from typing import TypedDict, Union

import networkx

from applications.Mapping import Mapping
from solvers.GreedyClassicalPVC import GreedyClassicalPVC
from solvers.ReverseGreedyClassicalPVC import ReverseGreedyClassicalPVC
from solvers.RandomClassicalPVC import RandomPVC


class Direct(Mapping):
    """
    Direct mapping. This usually means no significant mapping steps have to be done if any.
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.solver_options = ["GreedyClassicalPVC", "ReverseGreedyClassicalPVC", "RandomPVC"]

    def get_parameter_options(self) -> dict:
        """
        Returns empty dict as this mapping has no configurable settings

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

    def map(self, problem: networkx.Graph, config: dict) -> (networkx.Graph, float):
        """
        No mapping is required for this mapping.

        :param problem: networkx graph representing the problem
        :type  problem: networkx.Graph
        :param config: dict containing the config settings for the mapping
        :type config: Config
        :return: networkx graph representing the problem and the time it took to map it
        :rtype: tuple(networkx.Graph, float)
        """
        return problem, 0.0

    def get_solver(self, solver_option: str) -> Union[GreedyClassicalPVC, ReverseGreedyClassicalPVC, RandomPVC]:

        if solver_option == "GreedyClassicalPVC":
            return GreedyClassicalPVC()
        if solver_option == "ReverseGreedyClassicalPVC":
            return ReverseGreedyClassicalPVC()
        if solver_option == "RandomPVC":
            return RandomPVC()
        else:
            raise NotImplementedError(f"Solver Option {solver_option} not implemented")
