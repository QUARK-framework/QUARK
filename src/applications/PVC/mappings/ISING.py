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
from time import time

import networkx
import numpy as np
from dimod import qubo_to_ising

from applications.PVC.mappings.QUBO import Qubo
from solvers.PennylaneQAOA import PennylaneQAOA
from solvers.QAOA import QAOA
from applications.Mapping import *


class Ising(Mapping):
    """
    Ising formulation for the PVC

    """
    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.solver_options = ["QAOA", "PennylaneQAOA"]
        self.key_mapping = None

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

    def map(self, g: networkx.Graph, config: Config) -> (dict, float):
        """
        Uses the PVC QUBO formulation and converts it to an Ising.

        :param g: a networkx Graph
        :type g: networkx.Graph
        :param config: dictionary with the mapping config
        :type config: Config
        :return: dict with the ising, time it took to map it
        :rtype: tuple(dict, float)
        """
        start = time() * 1000
        qubo_mapping = Qubo()
        q, _ = qubo_mapping.map(g, config)
        t, j, _ = qubo_to_ising(q["Q"])

        config = [x[2]['c_start'] for x in g.edges(data=True)]
        config = list(set(config + [x[2]['c_end'] for x in g.edges(data=True)]))

        tool = [x[2]['t_start'] for x in g.edges(data=True)]
        tool = list(set(tool + [x[2]['t_end'] for x in g.edges(data=True)]))

        # Convert Ising dict to matrix
        timesteps = int((g.number_of_nodes() - 1) / 2 + 1)  # G.number_of_nodes()

        j_matrix = np.array(
            [[0.0] * g.number_of_nodes() * len(config) * len(tool) * timesteps for i in
             range(g.number_of_nodes() * timesteps * len(config) * len(tool))])

        self.key_mapping = {}
        index_counter = 0

        for key, value in j.items():
            if key[0] not in self.key_mapping:
                self.key_mapping[key[0]] = index_counter
                index_counter += 1
            if key[1] not in self.key_mapping:
                self.key_mapping[key[1]] = index_counter
                index_counter += 1
            u = self.key_mapping[key[0]]
            v = self.key_mapping[key[1]]
            j_matrix[u][v] = value

        return {"J": j_matrix, "t": np.array(list(t.values()))}, round(time() * 1000 - start, 3)

    def reverse_map(self, solution: dict) -> (dict, float):
        """
        Maps the solution back to the representation needed by the PVC class for validation/evaluation.

        :param solution: dictionary containing the solution
        :type solution: dict
        :return: solution mapped accordingly, time it took to map it
        :rtype: tuple(dict, float)
        """
        start = time() * 1000
        logging.info(f"Key Mapping: {self.key_mapping}")
        # TODO Maybe throw error here if solution contains too many 1's
        result = {}
        for key, value in self.key_mapping.items():
            result[key] = 1 if solution[value] == 1 else 0

        return result, round(time() * 1000 - start, 3)

    def get_solver(self, solver_option: str) -> Union[QAOA, PennylaneQAOA]:

        if solver_option == "QAOA":
            return QAOA()
        if solver_option == "PennylaneQAOA":
            return PennylaneQAOA()
        else:
            raise NotImplementedError(f"Solver Option {solver_option} not implemented")
