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
import logging

import dwave_networkx as dnx
import networkx

from quark.modules.applications.Mapping import Mapping, Core
from quark.utils import start_time_measurement, end_time_measurement


class QUBO(Mapping):
    """
    QUBO formulation for the TSP.
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__()
        self.submodule_options = ["Annealer"]

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module.

        :return: List of dict with requirements of this module
        """
        return [
            {"name": "networkx", "version": "3.4.2"},
            {"name": "dwave_networkx", "version": "0.8.15"}
        ]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this mapping.

        :return: Dictionary with configurable settings
        .. code-block:: python

            return {
                    "lagrange_factor": {
                                        "values": [0.75, 1.0, 1.25],
                                        "description": "By which factor would you like to multiply your "
                                                        "Lagrange?",
                                        "custom_input": True,
                                        "postproc": float
                    }
                }
        """
        return {
            "lagrange_factor": {
                "values": [0.75, 1.0, 1.25],
                "description": "By which factor would you like to multiply your Lagrange?",
                "custom_input": True,
                "allow_ranges": True,
                "postproc": float  # Since we allow custom input here we need to parse it to float (input is str)
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config.

        .. code-block:: python

             lagrange_factor: float

        """
        lagrange_factor: float

    def map(self, problem: networkx.Graph, config: Config) -> tuple[dict, float]:
        """
        Maps the networkx graph to a QUBO formulation.

        :param problem: Networkx graph
        :param config: Config with the parameters specified in Config class
        :return: Dict with QUBO, time it took to map it
        """
        start = start_time_measurement()
        lagrange = None
        lagrange_factor = config['lagrange_factor']
        weight = 'weight'

        # Taken from dwave_networkx.traveling_salesperson_qubo
        lagrange = problem.size(weight=weight) * problem.number_of_nodes() / problem.number_of_edges()

        lagrange = lagrange * lagrange_factor

        logging.info(f"Default Lagrange parameter: {lagrange}")

        # Get a QUBO representation of the problem
        q = dnx.traveling_salesperson_qubo(problem, lagrange, weight)

        return {"Q": q}, end_time_measurement(start)

    def get_default_submodule(self, option: str) -> Core:
        """
        Get the default submodule based on the given option.

        :param option: Submodule option
        :return: Corresponding submodule
        :raises NotImplemented: If the provided option is not implemented
        """

        if option == "Annealer":
            from quark.modules.solvers.Annealer import Annealer  # pylint: disable=C0415
            return Annealer()
        else:
            raise NotImplementedError(f"Solver Option {option} not implemented")
