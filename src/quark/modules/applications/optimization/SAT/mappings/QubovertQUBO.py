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

from quark.modules.applications.Mapping import Mapping, Core
from quark.utils import start_time_measurement, end_time_measurement


class QubovertQUBO(Mapping):
    """
    Qubovert formulation of the vehicle-options problem.
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__()
        self.submodule_options = ["Annealer"]
        self.pubo_problem = None
        self.nr_vars = None

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module.

        :return: List of dict with requirements of this module
        """
        return [
            {"name": "nnf", "version": "0.4.1"},
            {"name": "qubovert", "version": "1.2.5"}
        ]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this mapping.

        :return: Dict with configurable settings
        .. code-block:: python

            return {
                    "lagrange": {
                        "values": [0.1, 1, 1.5, 2, 5, 10, 1000, 10000],
                        "description": "By which factor would you like to multiply your Lagrange?"
                    }
                }
        """
        return {
            "lagrange": {
                "values": [0.1, 1, 1.5, 2, 5, 10, 1000, 10000],
                "description": "By which factor would you like to multiply your Lagrange?"
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config.

        .. code-block:: python

             lagrange: float

        """
        lagrange: float

    @staticmethod
    def _constraints2qubovert(constraints: any) -> AND:
        """
        Converts the constraints nnf to a PUBO in the qubovert library.

        :param constraints: Constraints in nnf format
        :return: Constraints in qubovert format
        """
        clauses = []
        for c in constraints.children:
            literals = [v.name if v.true else NOT(v.name) for v in c.children]
            clauses.append(OR(*literals))
        return AND(*clauses)

    @staticmethod
    def _tests2qubovert(test_clauses: dict) -> sum:
        """
        Converts the list of test clauses in the nnf format to a PUBO.

        :param test_clauses: Test clauses in nnf format
        :return: Sum of mapped test clauses
        """
        mapped_tests = []

        for test_clause in test_clauses:
            mapped_tests.append(OR(*[v.name if v.true else NOT(v.name) for v in test_clause.children]))

        return sum(mapped_tests)

    def map(self, problem: any, config: Config) -> tuple[dict, float]:
        """
        Converts the problem to a QUBO in dictionary format. Problem is a CNF formula from the nnf library.

        :param problem: SAT problem
        :param config: Config with the parameters specified in Config class
        :return: Dict with the QUBO, time it took to map it
        """
        start = start_time_measurement()
        lagrange = config['lagrange']

        constraints, test_clauses = problem

        # Find number of the variables that appear in the tests and constraints, to verify the reverse mapping.
        self.nr_vars = len(constraints.vars().union(And(test_clauses).vars()))

        # Convert the constraints to qubovert:
        constraints_pubo = self._constraints2qubovert(constraints)

        # Convert the tests into qubovert:
        tests_pubo = self._tests2qubovert(test_clauses)
        logging.info(f'{tests_pubo.to_qubo().num_terms} number of terms in tests qubo')
        lagrange *= len(test_clauses)

        # Define the total PUBO problem:
        self.pubo_problem = -(tests_pubo + lagrange * constraints_pubo)

        # Convert to qubo:
        qubo_problem = self.pubo_problem.to_qubo()
        qubo_problem.normalize()
        logging.info(f"Converted to QUBO with {qubo_problem.num_binary_variables} Variables."
                     f" Lagrange parameter: {config['lagrange']}.")

        # Convert it to the right format to be accepted by Braket / Dwave
        q_dict = {}

        for k, v in qubo_problem.items():
            # "interaction (quadratic) terms":
            if len(k) == 2:
                if (k[0], k[1]) not in q_dict:
                    q_dict[(k[0], k[1])] = float(v)
                else:
                    q_dict[(k[0], k[1])] += float(v)
            # "local (linear) fields":
            if len(k) == 1:
                if (k[0], k[0]) not in q_dict:
                    q_dict[(k[0], k[0])] = float(v)
                else:
                    q_dict[(k[0], k[0])] += float(v)

        return {"Q": q_dict}, end_time_measurement(start)

    def reverse_map(self, solution: dict) -> tuple[dict, float]:
        """
        Maps the solution back to the representation needed by the SAT class for validation/evaluation.

        :param solution: Dictionary containing the solution
        :return: Solution mapped accordingly, time it took to map it
        """
        start = start_time_measurement()
        pubo_sol = self.pubo_problem.convert_solution(solution)

        # Check if all variables appear in the solution.
        missing_vars = {f'L{i}' for i in range(self.nr_vars)} - set(pubo_sol.keys())

        # Add values for the missing variables -- if they do not appear, then their assignment does not matter.
        for missing_var in missing_vars:
            pubo_sol[missing_var] = True

        return pubo_sol, end_time_measurement(start)

    def get_default_submodule(self, option: str) -> Core:
        """
        Returns the default submodule based on the provided option.

        :param option: Option specifying the submodule
        :return: Instance of the corresponding submodule
        :raises NotImplementedError: If the option is not recognized
        """
        if option == "Annealer":
            from quark.modules.solvers.Annealer import Annealer  # pylint: disable=C0415
            return Annealer()
        else:
            raise NotImplementedError(f"Solver Option {option} not implemented")
