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

import networkx as nx
import pulser

from quark.modules.applications.Mapping import Mapping, Core
from quark.utils import start_time_measurement, end_time_measurement


class NeutralAtom(Mapping):
    """
    Neutral atom formulation for MIS.
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__()
        self.submodule_options = ["NeutralAtomMIS"]

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module.

        :return: List of requirements of this module
        """
        return [{"name": "pulser", "version": "1.1.1"}]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this mapping.

        :return: Empty dictionary, as this mapping has no configurable settings
        """
        return {}

    class Config(TypedDict):
        """
        Configuration options for Neutral Atom MIS mapping.
        """
        pass

    def map(self, problem: nx.Graph, config: Config) -> tuple[dict, float]:
        """
        Maps the networkx graph to a neutral atom MIS problem.

        :param problem: Networkx graph representing the MIS problem
        :param config: Config with the parameters specified in Config class
        :return: Tuple containing a dictionary with the neutral MIS and time it took to map it
        """
        start = start_time_measurement()

        pos = nx.get_node_attributes(problem, 'pos')
        register = pulser.Register(pos)

        neutral_atom_problem = {
            'graph': problem,
            'register': register
        }

        return neutral_atom_problem, end_time_measurement(start)

    def get_default_submodule(self, option: str) -> Core:
        """
        Returns the default submodule based on the provided option.

        :param option: Option specifying the submodule
        :return: Instance of the corresponding submodule
        :raises NotImplementedError: If the option is not recognized
        """
        if option == "NeutralAtomMIS":
            from quark.modules.solvers.NeutralAtomMIS import NeutralAtomMIS  # pylint: disable=C0415
            return NeutralAtomMIS()
        else:
            raise NotImplementedError(f"Solver Option {option} not implemented")
