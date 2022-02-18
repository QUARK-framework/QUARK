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
from typing import TypedDict
from qubovert.sat import NOT, OR, AND
from nnf import And
from applications.Mapping import *
from solvers.Annealer import Annealer


class QubovertQubo(Mapping):
    """
    Qubovert formulation of the vehicle-options problem.
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.solver_options = ["Annealer"]
        self.pubo_problem = None
        self.nr_vars = None

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this mapping

        :return:
                 .. code-block:: python

                     return {
                                "lagrange": {
                                    "values": [0.1, 1, 1.5, 2, 5, 10, 1000, 10000],
                                    "description": "What lagrange for the qubo mapping? 1 the number of tests."
                                }
                            }

        """
        return {
            "lagrange": {
                "values": [0.1, 1, 1.5, 2, 5, 10, 1000, 10000],
                "description": "What lagrange for the qubo mapping? 1 the number of tests."
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python

             lagrange: float

        """
        lagrange: float

    @staticmethod
    def _constraints2qubovert(constraints: dict) -> AND:
        """
        Converts the constraints nnf to a pubo in the qubovert library.

        :param constraints:
        :type constraints: dict
        :return:
        :rtype: AND
        """
        clauses = []
        for c in constraints.children:
            literals = [v.name if v.true else NOT(v.name) for v in c.children]
            clauses.append(OR(*literals))
        return AND(*clauses)

    @staticmethod
    def _tests2qubovert(test_clauses: dict) -> sum:
        """
        Converts the list of test clauses in the nnf format to a pubo.

        :param test_clauses:
        :type test_clauses: dict
        :return:
        :rtype: sum
        """
        mapped_tests = []

        for test_clause in test_clauses:
            mapped_tests.append(OR(*[v.name if v.true else NOT(v.name) for v in test_clause.children]))

        return sum(mapped_tests)

    def map(self, problem: any, config: Config) -> (dict, float):
        """
        Converts the problem to a Qubo in dictionary format. Problem is a CNF formula from the nnf library.

        :param problem:
        :type problem: any
        :param config: config with the parameters specified in Config class
        :type config: Config
        :return: dict with the QUBO, time it took to map it
        :rtype: tuple(dict, float)
        """
        start = time() * 1000
        lagrange = config['lagrange']

        constraints, test_clauses = problem

        # find number of the variables that appear in the tests and constraints, to verify the reverse mapping.
        self.nr_vars = len(constraints.vars().union(And(test_clauses).vars()))

        # first we convert the constraints to qubovert:
        constraints_pubo = self._constraints2qubovert(constraints)

        # next, we convert the tests into qubovert:
        tests_pubo = self._tests2qubovert(test_clauses)
        logging.info(f'{tests_pubo.to_qubo().num_terms} number of terms in tests qubo')
        lagrange *= len(test_clauses)

        # define the total pubo problem:
        self.pubo_problem = -(tests_pubo + lagrange * constraints_pubo)
        # convert to qubo:
        qubo_problem = self.pubo_problem.to_qubo()
        qubo_problem.normalize()
        logging.info(f"Converted to QUBO with {qubo_problem.num_binary_variables} Variables."
                     f" Lagrange parameter: {config['lagrange']}.")

        # now we need to convert it to the right format to be accepted by Braket / Dwave
        q_dict = {}

        for k, v in qubo_problem.items():
            # "interaction (quadratic) terms":
            if len(k) == 2:
                if (k[0], k[1]) not in q_dict.keys():
                    q_dict[(k[0], k[1])] = float(v)
                else:
                    q_dict[(k[0], k[1])] += float(v)
            # "local (linear) fields":
            if len(k) == 1:
                if (k[0], k[0]) not in q_dict.keys():
                    q_dict[(k[0], k[0])] = float(v)
                else:
                    q_dict[(k[0], k[0])] += float(v)

        return {"Q": q_dict}, round(time() * 1000 - start, 3)

    def reverse_map(self, solution: dict) -> (dict, float):
        """
        Maps the solution back to the representation needed by the SAT class for validation/evaluation.

        :param solution: dictionary containing the solution
        :type solution: dict
        :return: solution mapped accordingly, time it took to map it
        :rtype: tuple(dict, float)
        """
        start = time() * 1000
        pubo_sol = self.pubo_problem.convert_solution(solution)
        # Let's check if all variables appear in the solution.
        missing_vars = {f'L{i}' for i in range(self.nr_vars)} - set(pubo_sol.keys())
        # add values for the missing variables -- if they do not appear, then their assignment does not matter.
        for missing_var in missing_vars:
            pubo_sol[missing_var] = True
        return pubo_sol, round(time() * 1000 - start, 3)

    def get_solver(self, solver_option: str) -> Annealer:

        if solver_option == "Annealer":
            return Annealer()
        else:
            raise NotImplementedError(f"Solver Option {solver_option} not implemented")
