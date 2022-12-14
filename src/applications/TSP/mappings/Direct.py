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

from applications.Mapping import *
from solvers.GreedyClassicalTSP import GreedyClassicalTSP
from solvers.ReverseGreedyClassicalTSP import ReverseGreedyClassicalTSP
from solvers.RandomClassicalTSP import RandomTSP


class Direct(Mapping):
    """
    Direct mapping. This usually means no significant mapping steps have to be done if any.
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.solver_options = ["GreedyClassicalTSP", "ReverseGreedyClassicalTSP", "RandomTSP"]

    def get_parameter_options(self) -> dict:
        """
        Returns empty dict as this mapping has no configurable settings.

        :return: empty dict
        :rtype: dict
        """
        return {

        }

    class Config(TypedDict):
        """
        Empty config as this solver has no configurable settings.
        """
        pass

    def map(self, problem: networkx.Graph, config: Config) -> (networkx.Graph, float):
        """
        No mapping is required here.

        :param problem: networkx graph
        :type problem: networkx.Graph
        :param config: config with the parameters specified in Config class
        :param config: Config
        :return: networkx graph, time it took to map it
        :rtype: tuple(networkx.Graph, float)
        """
        return problem, 0.0

    def get_solver(self, solver_option: str) -> Union[GreedyClassicalTSP, ReverseGreedyClassicalTSP, RandomTSP]:

        if solver_option == "GreedyClassicalTSP":
            return GreedyClassicalTSP()
        if solver_option == "ReverseGreedyClassicalTSP":
            return ReverseGreedyClassicalTSP()
        if solver_option == "RandomTSP":
            return RandomTSP()
        else:
            raise NotImplementedError(f"Solver Option {solver_option} not implemented")
