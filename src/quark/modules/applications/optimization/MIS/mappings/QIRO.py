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

from quark.modules.applications.Mapping import Core, Mapping
from quark.utils import start_time_measurement, end_time_measurement


class QIRO(Mapping):
    """
    The quantum-informed recursive optimization (QIRO) formulation for the MIS problem. QIRO recursively simplifies the
    problem classically using information obtained with quantum resources.
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.submodule_options = ["QrispQIRO"]

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module.

        :return: list of dict with requirements of this module
        """
        return [{"name": "qrisp", "version": "0.5.2"}]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this mapping.

        :return:
        .. code-block:: python

            return {}

        """
        # TODO "optimizer": {
        #          "values": ["not", "yet", "implemented"],
        #          "description": "Which QIRO algorithm should be used?"
        #      }
        return {}

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python
            pass
        """
        pass

    def map(self, problem: networkx.Graph, config: Config) -> tuple[dict, float]:
        """
        Maps the networkx graph to a neutral atom MIS problem.

        :param problem: Networkx graph
        :param config: Config with the parameters specified in Config class
        :return: Dict with neutral MIS, time it took to map it
        """
        start = start_time_measurement()

        qiro_mapped_problem = {
            'graph': problem,
        }
        return qiro_mapped_problem, end_time_measurement(start)

    def get_default_submodule(self, option: str) -> Core:
        """
        Returns the default submodule based on the provided option.

        :param option: Option specifying the submodule
        :return: Instance of the corresponding submodule
        :raises NotImplementedError: If the option is not recognized
        """
        if option == "QrispQIRO":
            from quark.modules.solvers.QrispQIRO import QIROSolver  # pylint: disable=C0415
            return QIROSolver()
        else:
            raise NotImplementedError(f"Solver Option {option} not implemented")
