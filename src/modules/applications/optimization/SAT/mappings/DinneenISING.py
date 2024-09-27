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

import numpy as np
from dimod import qubo_to_ising
from nnf import And

from modules.applications.Mapping import *
from modules.applications.optimization.SAT.mappings.DinneenQUBO import DinneenQUBO
from utils import start_time_measurement, end_time_measurement


class DinneenIsing(Mapping):
    """
    Ising formulation for SAT using Dinneen QUBO.

    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.submodule_options = ["QAOA", "PennylaneQAOA"]
        self.problem = None
        self.qubo_mapping = None

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module

        :return: list of dict with requirements of this module
        :rtype: list[dict]
        """
        return [
            {
                "name": "nnf",
                "version": "0.4.1"
            },
            {
                "name": "numpy",
                "version": "1.26.4"
            },
            {
                "name": "dimod",
                "version": "0.12.17"
            },
            *DinneenQUBO.get_requirements()
        ]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this mapping

        :return:
                 .. code-block:: python

                     return {
                                "lagrange": {
                                    "values": [0.1, 1, 2],
                                    "description": "What lagrange parameter to multiply with the number of (hard) "
                                                   "constraints?"
                                }
                            }

        """
        return {
            "lagrange": {
                "values": [0.1, 1, 2],
                "description": "What lagrange parameter to multiply with the number of (hard) constraints?"
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python

             lagrange: float

        """
        lagrange: float

    def map(self, problem: any, config) -> (dict, float):
        """
        Uses the DinneenQUBO formulation and converts it to an Ising.

        :param problem: the SAT problem
        :type problem: any
        :param config: dictionary with the mapping config
        :type config: Config
        :return: dict with the ising, time it took to map it
        :rtype: tuple(dict, float)
        """
        start = start_time_measurement()
        self.problem = problem
        # call mapping function
        self.qubo_mapping = DinneenQUBO()
        q, _ = self.qubo_mapping.map(problem, config)
        t, j, _ = qubo_to_ising(q["Q"])

        # Convert Ising dict to matrix
        n = (len(problem[0]) + len(problem[1])) + len(problem[0].vars().union(And(problem[1]).vars()))
        t_vector = np.zeros(n, dtype=float)
        j_matrix = np.zeros((n, n), dtype=float)

        for key, value in t.items():
            t_vector[key] = value

        for key, value in j.items():
            j_matrix[key[0]][key[1]] = value

        return {"J": j_matrix, "t": t_vector}, end_time_measurement(start)

    def reverse_map(self, solution: dict) -> (dict, float):
        """
        Maps the solution back to the representation needed by the SAT class for validation/evaluation.

        :param solution: dictionary containing the solution
        :type: dict
        :return: solution mapped accordingly, time it took to map it
        :rtype: tuple(dict, float)
        """
        start = start_time_measurement()
        # convert raw solution into the right format to use reverse_map() of ChoiQUBO.py
        solution_dict = {}
        for i, el in enumerate(solution):
            solution_dict[i] = el

        # reverse map
        result, _ = self.qubo_mapping.reverse_map(solution_dict)

        return result, end_time_measurement(start)

    def get_default_submodule(self, option: str) -> Core:

        if option == "QAOA":
            from modules.solvers.QAOA import QAOA  # pylint: disable=C0415
            return QAOA()
        if option == "PennylaneQAOA":
            from modules.solvers.PennylaneQAOA import PennylaneQAOA  # pylint: disable=C0415
            return PennylaneQAOA()
        else:
            raise NotImplementedError(f"Solver Option {option} not implemented")
