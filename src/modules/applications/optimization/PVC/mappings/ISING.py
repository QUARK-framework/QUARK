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
import numpy as np
from dimod import qubo_to_ising

from modules.applications.Mapping import *
from modules.applications.optimization.PVC.mappings.QUBO import Qubo


class Ising(Mapping):
    """
    Ising formulation for the PVC

    """
    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.submodule_options = ["QAOA", "PennylaneQAOA"]
        self.key_mapping = None

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
                "version": "2.8.8"
            },
            {
                "name": "numpy",
                "version": "1.24.1"
            },
            {
                "name": "dimod",
                "version": "0.12.5"
            },
            *Qubo.get_requirements()
        ]

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

    def map(self, problem: networkx.Graph, config: Config) -> (dict, float):
        """
        Uses the PVC QUBO formulation and converts it to an Ising.

        :param problem: a networkx Graph
        :type problem: networkx.Graph
        :param config: dictionary with the mapping config
        :type config: Config
        :return: dict with the ising, time it took to map it
        :rtype: tuple(dict, float)
        """
        start = time() * 1000
        qubo_mapping = Qubo()
        q, _ = qubo_mapping.map(problem, config)
        t, j, _ = qubo_to_ising(q["Q"])

        config = [x[2]['c_start'] for x in problem.edges(data=True)]
        config = list(set(config + [x[2]['c_end'] for x in problem.edges(data=True)]))

        tool = [x[2]['t_start'] for x in problem.edges(data=True)]
        tool = list(set(tool + [x[2]['t_end'] for x in problem.edges(data=True)]))

        # Convert Ising dict to matrix
        timesteps = int((problem.number_of_nodes() - 1) / 2 + 1)  # G.number_of_nodes()

        matrix_size = problem.number_of_nodes() * len(config) * len(tool) * timesteps
        j_matrix = np.zeros((matrix_size, matrix_size), dtype=float)

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

    def get_default_submodule(self, option: str) -> Core:

        if option == "QAOA":
            from modules.solvers.QAOA import QAOA  # pylint: disable=C0415
            return QAOA()
        if option == "PennylaneQAOA":
            from modules.solvers.PennylaneQAOA import PennylaneQAOA  # pylint: disable=C0415
            return PennylaneQAOA()
        else:
            raise NotImplementedError(f"Solver Option {option} not implemented")
