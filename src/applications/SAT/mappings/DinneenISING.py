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

import numpy as np
from dimod import qubo_to_ising

from nnf import And
from applications.SAT.mappings.DinneenQUBO import DinneenQubo
from solvers.PennylaneQAOA import PennylaneQAOA
from solvers.QAOA import QAOA
from applications.Mapping import *


class DinneenIsing(Mapping):
    """
    Ising formulation for SAT using Dinneen QUBO.

    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.solver_options = ["QAOA", "PennylaneQAOA"]
        self.problem = None
        self.qubo_mapping = None

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this mapping

        :return:
                 .. code-block:: python

                     return {
                                "lagrange": {
                                    "values": [0.1, 1, 2],
                                    "description": "What lagrange parameter to multiply with the number of (hard) constraints?"
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
        start = time() * 1000
        self.problem = problem
        # call mapping function
        self.qubo_mapping = DinneenQubo()
        q, _ = self.qubo_mapping.map(problem, config)
        t, j, _ = qubo_to_ising(q["Q"])

        # Convert Ising dict to matrix
        n = (len(problem[0]) + len(problem[1])) + len(problem[0].vars().union(And(problem[1]).vars()))
        t_vector = np.array([0.0] * n)
        j_matrix = np.array([[0.0] * n for _ in range(n)])

        for key, value in t.items():
            t_vector[key] = value

        for key, value in j.items():
            j_matrix[key[0]][key[1]] = value

        return {"J": j_matrix, "t": t_vector}, round(time() * 1000 - start, 3)

    def reverse_map(self, solution: dict) -> (dict, float):
        """
        Maps the solution back to the representation needed by the SAT class for validation/evaluation.

        :param solution: dictionary containing the solution
        :type: dict
        :return: solution mapped accordingly, time it took to map it
        :rtype: tuple(dict, float)
        """
        start = time() * 1000
        # convert raw solution into the right format to use reverse_map() of ChoiQUBO.py
        solution_dict = {}
        for i, el in enumerate(solution):
            solution_dict[i] = el

        # reverse map
        result, _ = self.qubo_mapping.reverse_map(solution_dict)

        return result, round(time() * 1000 - start, 3)

    def get_solver(self, solver_option: str) -> Union[QAOA, PennylaneQAOA]:

        if solver_option == "QAOA":
            return QAOA()
        if solver_option == "PennylaneQAOA":
            return PennylaneQAOA()
        else:
            raise NotImplementedError(f"Solver Option {solver_option} not implemented")
