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

import logging
from typing import TypedDict, Union
import dwave_networkx as dnx
import networkx

from applications.Mapping import *
from solvers.Annealer import Annealer


class Qubo(Mapping):
    """
    QUBO formulation for the TSP.

    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.solver_options = ["Annealer"]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this mapping

        :return:
                 .. code-block:: python

                     return {
                                "lagrange_factor": {
                                    "values": [0.75, 1.0, 1.25],
                                    "description": "By which factor would you like to multiply your lagrange?"
                                }
                            }

        """
        return {
            "lagrange_factor": {
                "values": [0.75, 1.0, 1.25],
                "description": "By which factor would you like to multiply your lagrange?"
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python

             lagrange_factor: float

        """
        lagrange_factor: float

    def map(self, graph: networkx.Graph, config: Config) -> (dict, float):
        """
        Maps the networkx graph to a QUBO formulation.

        :param graph: networkx graph
        :type graph: networkx.Graph
        :param config: config with the parameters specified in Config class
        :type config: Config
        :return: dict with QUBO, time it took to map it
        :rtype: tuple(dict, float)
        """
        start = time() * 1000
        lagrange = None
        lagrange_factor = config['lagrange_factor']
        weight = 'weight'

        if lagrange is None:
            # If no lagrange parameter provided, set to 'average' tour length.
            # Usually a good estimate for a lagrange parameter is between 75-150%
            # of the objective function value, so we come up with an estimate for
            # tour length and use that.
            if graph.number_of_edges() > 0:
                lagrange = graph.size(weight=weight) * graph.number_of_nodes() / graph.number_of_edges()
            else:
                lagrange = 2

        lagrange = lagrange * lagrange_factor

        logging.info(f"Default Lagrange parameter: {lagrange}")

        # Get a QUBO representation of the problem
        q = dnx.traveling_salesperson_qubo(graph, lagrange, weight)

        return {"Q": q}, round(time() * 1000 - start, 3)

    def get_solver(self, solver_option: str) -> Union[Annealer]:

        if solver_option == "Annealer":
            return Annealer()
        else:
            raise NotImplementedError(f"Solver Option {solver_option} not implemented")
