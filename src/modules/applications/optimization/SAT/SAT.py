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
from typing import TypedDict, List, Dict, Tuple, Any

import nnf
import numpy as np
from nnf import Var, And, Or
from nnf.dimacs import dump

from modules.applications.Application import Application
from modules.applications.optimization.Optimization import Optimization
from utils import start_time_measurement, end_time_measurement


class SAT(Optimization):
    """
    Before a new vehicle model can be deployed for production, several tests have to be carried out on pre-series
    vehicles to ensure the feasibility and gauge the functionality of specific configurations of components.
    Naturally, the manufacturer wants to save resources and produce as few pre-series vehicles as possible while
    still performing all desired tests. Further, not all feature configurations can realistically be implemented in
    all vehicles, leading to constraints that the produced vehicles must satisfy. This can be modeled as a SAT problem.
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__("SAT")
        self.submodule_options = [
            "QubovertQUBO", "Direct", "ChoiQUBO", "DinneenQUBO", "ChoiIsing", "DinneenIsing"
        ]
        self.literals = None
        self.num_tests = None
        self.num_constraints = None
        self.num_variables = None

    @staticmethod
    def get_requirements() -> List[Dict]:
        """
        Return requirements of this module.

        :return: list of dict with requirements of this module
        """
        return [
            {"name": "nnf", "version": "0.4.1"},
            {"name": "numpy", "version": "1.26.4"}
        ]

    def get_solution_quality_unit(self) -> str:
        return "Evaluation"

    def get_default_submodule(self, option: str) -> Application:

        if option == "QubovertQUBO":
            from modules.applications.optimization.SAT.mappings.QubovertQUBO import \
                QubovertQUBO  # pylint: disable=C0415
            return QubovertQUBO()
        elif option == "Direct":
            from modules.applications.optimization.SAT.mappings.Direct import Direct  # pylint: disable=C0415
            return Direct()
        elif option == "ChoiQUBO":
            from modules.applications.optimization.SAT.mappings.ChoiQUBO import ChoiQUBO  # pylint: disable=C0415
            return ChoiQUBO()
        elif option == "ChoiIsing":
            from modules.applications.optimization.SAT.mappings.ChoiISING import ChoiIsing  # pylint: disable=C0415
            return ChoiIsing()
        elif option == "DinneenQUBO":
            from modules.applications.optimization.SAT.mappings.DinneenQUBO import DinneenQUBO  # pylint: disable=C0415
            return DinneenQUBO()
        elif option == "DinneenIsing":
            from modules.applications.optimization.SAT.mappings.DinneenISING import \
                DinneenIsing  # pylint: disable=C0415
            return DinneenIsing()
        else:
            raise NotImplementedError(f"Mapping Option {option} not implemented")

    def get_parameter_options(self) -> Dict:
        """
        Returns the configurable settings for this application

        :return: Dict with cnfigurable settings
                 .. code-block:: python

                      return {
                                "variables": {
                                    "values": list(range(10, 151, 10)),
                                    "custom_input": True,
                                    "allow_ranges": True,
                                    "postproc": int,
                                    "description": "How many variables do you need?"
                                },
                                "clvar_ratio_cons": {
                                    "values": [2, 3, 4, 4.2, 5],
                                    "custom_input": True,
                                    "allow_ranges": True,
                                    "postproc": int,
                                    "description": "What clause:variable ratio do you want for the (hard) constraints?"
                                },
                                "clvar_ratio_test": {
                                    "values": [2, 3, 4, 4.2, 5],
                                    "custom_input": True,
                                    "allow_ranges": True,
                                    "postproc": int,
                                    "description": "What clause:variable ratio do you want for the tests (soft con.)?"
                                },
                                "problem_set": {
                                    "values": list(range(10)),
                                    "description": "Which problem set do you want to use?"
                                },
                                "max_tries": {
                                    "values": [100],
                                    "description": "Maximum number of tries to create problem"
                                }
                            }

        """
        return {
            "variables": {
                "values": list(range(10, 101, 10)),
                "custom_input": True,
                "allow_ranges": True,
                "postproc": int,
                "description": "How many variables do you need?"
            },
            "clvar_ratio_cons": {
                "values": [2, 3, 4, 4.2, 5],
                "custom_input": True,
                "allow_ranges": True,
                "postproc": int,
                "description": "What clause:variable ratio do you want for the (hard) constraints?"
            },
            "clvar_ratio_test": {
                "values": [2, 3, 4, 4.2, 5],
                "custom_input": True,
                "allow_ranges": True,
                "postproc": int,
                "description": "What clause:variable ratio do you want for the tests (soft constraints)?"
            },
            "problem_set": {
                "values": list(range(10)),
                "description": "Which problem set do you want to use?"
            },
            "max_tries": {
                "values": [100],
                "description": "Maximum number of tries to create problem"
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python

            variables: int
            clvar_ratio_cons: float
            clvar_ratio_test: float
            problem_set: int
            max_tries: int

        """
        variables: int
        clvar_ratio_cons: float
        clvar_ratio_test: float
        problem_set: int
        max_tries: int

    def generate_problem(self, config: Config) -> Tuple[nnf.And, list]:
        """
        Generates a vehicle configuration problem out of a given config. Returns buildability constraints (hard
        constraints) and tests (soft constraints), the successful evaluation of which we try to maximize. Both
        are given in nnf form, which we then convert accordingly.

        :param config: Configuration parameters for problem generation
        :return: A tuple containing the problem, number of variables, and other details
        """

        self.num_variables = config["variables"]
        num_constraints = round(config["clvar_ratio_cons"] * self.num_variables)
        num_tests = round(config["clvar_ratio_test"] * self.num_variables)

        max_tries = config["max_tries"]
        self.literals = [Var(f"L{i}") for i in range(self.num_variables)]

        self.application = {}

        def _generate_3sat_clauses(nr_clauses, nr_vars, satisfiable, rseed, nr_tries):
            # iterate over the desired number of attempts: break if we find a solvable instance.
            for attempt in range(nr_tries):
                # initialize random number generator -- multiply the attempt to traverse distinct random seeds
                # for the hard and soft constraints, respectively (since rseed of the hard and soft constraints differs
                # by 1).
                rng = np.random.default_rng(rseed + attempt * 2)
                clause_list = []
                # generate literal list to sample from
                lit_vars = [Var(f"L{i}") for i in range(nr_vars)]
                for _ in range(nr_clauses):
                    # Select three (non-repeated) literals and negate them randomly -- together constituting a clause
                    chosen_literals = rng.choice(lit_vars, 3, replace=False)
                    negate_literals = rng.choice([True, False], 3, replace=True)
                    clause = []
                    # Perform the random negations and append to clause:
                    for lit, neg in zip(chosen_literals, negate_literals):
                        if neg:
                            clause.append(lit.negate())
                        else:
                            clause.append(lit)
                    # Append the generated clause to the total container
                    clause_list.append(Or(clause))
                prob = And(clause_list)

                if not satisfiable or prob.satisfiable():
                    return clause_list

            # loop ran out of tries
            logging.error("Unable to generate valid solutions. Consider increasing max_tries or decreasing "
                          "the clause:variable ratio.")
            raise ValueError("Unable to generate valid solution.")

        # we choose a random seed -- since we try at most max_tries times to generate a solvable instance,
        # Space the initial random seeds by 2 * max_tries (because we need both hard and soft constraints).
        random_seed = 2 * config["problem_set"] * max_tries
        # generate hard  & soft constraints. Make both satisfiable, but this can in principle be tuned.
        hard = And(_generate_3sat_clauses(num_constraints, self.num_variables,
                                          satisfiable=True, rseed=random_seed, nr_tries=max_tries))
        # the random_seed + 1 ensures that a different set of seeds is sampled compared to the hard constraints.
        soft = _generate_3sat_clauses(num_tests, self.num_variables, satisfiable=True, rseed=random_seed + 1,
                                      nr_tries=config["max_tries"])
        if (hard is None) or (soft is None):
            raise ValueError("Unable to generate satisfiable")
        # saving constraints and tests
        self.application["constraints"] = hard
        self.application["tests"] = soft
        # and their cardinalities:
        self.num_constraints = len(hard)
        self.num_tests = len(soft)

        logging.info(f"Generated a vehicle options Max3SAT"
                     f" instance with {self.num_variables} variables, {self.num_constraints} constraints"
                     f" and {self.num_tests} tests")
        return hard, soft

    def validate(self, solution: Dict) -> Tuple[bool, float]:
        """
        Validate a given solution against the constraints.

        :param solution: The solution to validate
        :return: True if the solution is valid, False otherwise, time it took to complete
        """
        start = start_time_measurement()

        logging.info("Checking validity of solution:")
        nr_satisfied_hardcons = len(*np.where(
            [c.satisfied_by(solution) for c in self.application["constraints"].children]
        ))
        ratio = nr_satisfied_hardcons / self.num_constraints
        is_valid = ratio == 1.0
        # prints the ratio of satisfied constraints and prints if all constraints are satisfied
        logging.info(f"Ratio of satisfied constraints: {ratio}\nSuccess:{['no', 'yes'][int(is_valid)]}")

        return is_valid, end_time_measurement(start)

    def evaluate(self, solution: Dict) -> Tuple[float, float]:
        """
        Calculates the quality of the solution.

        :param solution: dictionary containing the solution
        :return: Tour length, time it took to calculate the tour length
        """
        start = start_time_measurement()
        logging.info("Checking the quality of the solution:")

        # count the number of satisfied clauses
        nr_satisfied_tests = len(*np.where([test.satisfied_by(solution) for test in self.application["tests"]]))

        ratio_satisfied = nr_satisfied_tests / self.num_tests
        logging.info(f"Ratio of satisfied test clauses: {ratio_satisfied}.")

        return ratio_satisfied, end_time_measurement(start)

    def save(self, path: str, iter_count: int) -> None:
        """
        Save the constraints and tests to files in CNF format.

        :param path: The directory path where the files will be saved.
        :param iter_count: The iteration count to include in the filenames.
        """
        with open(f"{path}/constraints_iter_{iter_count}.cnf", "w") as f_cons:
            dump(
                obj=self.application["constraints"],
                fp=f_cons,
                var_labels={str(literal): idx + 1 for idx, literal in enumerate(self.literals)}
            )
        with open(f"{path}/tests_iter_{iter_count}.cnf", "w") as f_test:
            dump(
                obj=Or(self.application["tests"]),
                fp=f_test,
                var_labels={str(literal): idx + 1 for idx, literal in enumerate(self.literals)}
            )
