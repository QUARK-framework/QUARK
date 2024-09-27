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

import dwave_networkx as dnx
import networkx

from modules.applications.Mapping import *
from utils import start_time_measurement, end_time_measurement


class QUBO(Mapping):
    """
    QUBO formulation for the TSP.

    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.submodule_options = ["Annealer"]

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module

        :return: list of dict with requirements of this module
        :rtype: list[dict]
        """
        return [
            {
                "name": "networkx",
                "version": "3.2.1"
            },
            {
                "name": "dwave_networkx",
                "version": "0.8.15"
            }
        ]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this mapping

        :return:
                 .. code-block:: python

                     return {
                                "lagrange_factor": {
                                                    "values": [0.75, 1.0, 1.25],
                                                    "description": "By which factor would you like to multiply your "
                                                                    "lagrange?",
                                                    "custom_input": True,
                                                    "postproc": float
                                }
                            }

        """
        return {
            "lagrange_factor": {
                "values": [0.75, 1.0, 1.25],
                "description": "By which factor would you like to multiply your lagrange?",
                "custom_input": True,
                "allow_ranges": True,
                "postproc": float  # Since we allow custom input here we need to parse it to float (input is str)
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python

             lagrange_factor: float

        """
        lagrange_factor: float

    def map(self, problem: networkx.Graph, config: Config) -> (dict, float):
        """
        Maps the networkx graph to a QUBO formulation.

        :param problem: networkx graph
        :type problem: networkx.Graph
        :param config: config with the parameters specified in Config class
        :type config: Config
        :return: dict with QUBO, time it took to map it
        :rtype: tuple(dict, float)
        """
        start = start_time_measurement()
        lagrange = None
        lagrange_factor = config['lagrange_factor']
        weight = 'weight'
        # get corresponding QUBO step by step

        if lagrange is None:
            # If no lagrange parameter provided, set to 'average' tour length.
            # Usually a good estimate for a lagrange parameter is between 75-150%
            # of the objective function value, so we come up with an estimate for
            # tour length and use that.
            if problem.number_of_edges() > 0:
                lagrange = problem.size(weight=weight) * problem.number_of_nodes() / problem.number_of_edges()
            else:
                lagrange = 2

        lagrange = lagrange * lagrange_factor

        logging.info(f"Default Lagrange parameter: {lagrange}")

        # Get a QUBO representation of the problem
        q = dnx.traveling_salesperson_qubo(problem, lagrange, weight)

        return {"Q": q}, end_time_measurement(start)

    def get_default_submodule(self, option: str) -> Core:

        if option == "Annealer":
            from modules.solvers.Annealer import Annealer  # pylint: disable=C0415
            return Annealer()
        else:
            raise NotImplementedError(f"Solver Option {option} not implemented")
