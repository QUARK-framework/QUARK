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
import nnf
from nnf import Var, And, Or
from nnf.dimacs import dump
import numpy as np

from applications.Application import *
from applications.SAT.mappings.Direct import Direct
from applications.SAT.mappings.QubovertQUBO import QubovertQubo
from applications.SAT.mappings.ChoiQUBO import ChoiQubo
from applications.SAT.mappings.DinneenQUBO import DinneenQubo
from applications.SAT.mappings.ChoiISING import ChoiIsing
from applications.SAT.mappings.DinneenISING import DinneenIsing


class SAT(Application):
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
        self.mapping_options = ["QubovertQubo", "Direct", "ChoiQubo", "DinneenQubo", "ChoiIsing", "DinneenIsing"]
        self.literals = None
        self.num_tests = None
        self.num_constraints = None
        self.num_variables = None

    def get_solution_quality_unit(self) -> str:
        return "Evaluation"

    def get_mapping(self, mapping_option: str) -> Union[QubovertQubo, Direct, ChoiQubo, DinneenQubo, DinneenIsing, ChoiIsing]:

        if mapping_option == "QubovertQubo":
            return QubovertQubo()
        elif mapping_option == "Direct":
            return Direct()
        elif mapping_option == 'ChoiQubo':
            return ChoiQubo()
        elif mapping_option == 'ChoiIsing':
            return ChoiIsing()
        elif mapping_option == 'DinneenQubo':
            return DinneenQubo()
        elif mapping_option == 'DinneenIsing':
            return DinneenIsing()
        else:
            raise NotImplementedError(f"Mapping Option {mapping_option} not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this application

        :return:
                 .. code-block:: python

                      return {
                                "variables": {
                                    "values": list(range(10, 151, 10)),
                                    "description": "How many variables do you need?"
                                },
                                "clvar_ratio_cons": {
                                    "values": [2, 3, 4, 4.2, 5],
                                    "description": "What clause:variable ratio do you want for the (hard) constraints?"
                                },
                                "clvar_ratio_test": {
                                    "values": [2, 3, 4, 4.2, 5],
                                    "description": "What clause:variable ratio do you want for the tests (soft constraints)?"
                                },
                                "problem_set": {
                                    "values": list(range(10)),
                                    "description": "Which problem set do you want to use?"
                                },
                                "max_tries": {
                                    "values": [100],
                                    "description": "Number of maximum tries"
                                }
                            }

        """
        return {
            "variables": {
                "values": list(range(10, 151, 10)),
                "description": "How many variables do you need?"
            },
            "clvar_ratio_cons": {
                "values": [2, 3, 4, 4.2, 5],
                "description": "What clause:variable ratio do you want for the (hard) constraints?"
            },
            "clvar_ratio_test": {
                "values": [2, 3, 4, 4.2, 5],
                "description": "What clause:variable ratio do you want for the tests (soft constraints)?"
            },
            "problem_set": {
                "values": list(range(10)),
                "description": "Which problem set do you want to use?"
            },
            "max_tries": {
                "values": [100],
                "description": "Number of maximum tries"
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

    def generate_problem(self, config: Config, iter_count: int) -> (nnf.And, list):
        """
        Generates a vehicle configuration problem out of a given config. Returns buildability constraints (hard
        constraints) and tests (soft constraints), the successful evaluation of which we try to maximize. Both
        are given in nnf form, which we then convert accordingly.

        :param config: config with the parameters specified in Config class
        :type config: Config
        :param iter_count: the iteration count
        :type iter_count: int
        :return:
        :rtype: tuple(nnf.And, list)
        """

        self.num_variables = config['variables']
        num_constraints = round(config['clvar_ratio_cons'] * self.num_variables)
        num_tests = round(config['clvar_ratio_test'] * self.num_variables)

        max_tries = config['max_tries']
        self.literals = [Var(f'L{i}') for i in range(self.num_variables)]

        self.application = dict()

        def _generate_3sat_clauses(nr_clauses, nr_vars, satisfiable, rseed, nr_tries):
            # iterate over the desired number of attempts: we break if we find a solvable instance.
            for attempt in range(nr_tries):
                # initialize random number generator -- we multiply the attempt to traverse distinct random seeds
                # for the hard and soft constraints, respectively (since rseed of the hard and soft constraints differs
                # by 1).
                rng = np.random.default_rng(rseed + attempt * 2)
                clause_list = []
                # generate literal list to sample from
                lit_vars = [Var(f'L{i}') for i in range(nr_vars)]
                for _ in range(nr_clauses):
                    # we select three (non-repeated) literals and negate them randomly -- together constituting a clause
                    chosen_literals = rng.choice(lit_vars, 3, replace=False)
                    negate_literals = rng.choice([True, False], 3, replace=True)
                    clause = []
                    # we perform the random negations and append to clause:
                    for lit, neg in zip(chosen_literals, negate_literals):
                        if neg:
                            clause.append(lit.negate())
                        else:
                            clause.append(lit)
                    # append the generated clause to the total container
                    clause_list.append(Or(clause))
                # we generate the conjunction of the problem, such that we can use the nnf native function and test its
                # satisfiability.
                prob = And(clause_list)

                if satisfiable and not prob.satisfiable():
                    if attempt == nr_tries - 1:
                        logging.error("Unable to generate valid solutions. Consider increasing max_tries or decreasing "
                                      "the clause:variable ratio.")
                        raise ValueError("Unable to generate valid solution.")
                    else:
                        continue
                else:
                    return clause_list

        # we choose a random seed -- since we try at most max_tries times to generate a solvable instance,
        # we space the initial random seeds by 2 * max_tries (because we need both hard and soft constraints).
        random_seed = 2 * config['problem_set'] * max_tries
        # generate hard  & soft constraints. We make both satisfiable, but this can in principle be tuned.
        hard = And(_generate_3sat_clauses(num_constraints, self.num_variables,
                                          satisfiable=True, rseed=random_seed, nr_tries=max_tries))
        # the random_seed + 1 ensures that a different set of seeds is sampled compared to the hard constraints.
        soft = _generate_3sat_clauses(num_tests, self.num_variables, satisfiable=True, rseed=random_seed + 1,
                                      nr_tries=config['max_tries'])
        if (hard is None) or (soft is None):
            raise ValueError("Unable to generate satisfiable")
        # saving constraints and tests
        self.application['constraints'] = hard
        self.application['tests'] = soft
        # and their cardinalities:
        self.num_constraints = len(hard)
        self.num_tests = len(soft)

        logging.info(f'Generated a vehicle options Max3SAT'
                     f' instance with {self.num_variables} variables, {self.num_constraints} constraints'
                     f' and {self.num_tests} tests')
        return hard, soft

    def validate(self, solution: dict) -> (bool, float):
        """
        Checks given solution.

        :param solution:
        :type solution: dict
        :return: Boolean whether the solution is valid, time it took to validate
        :rtype: tuple(bool, float)
        """
        start = time() * 1000

        logging.info("Checking validity of solution:")
        # logging.info(solution)
        nr_satisfied_hardcons = len(*np.where(
            [c.satisfied_by(solution) for c in self.application['constraints'].children]
        ))
        ratio = nr_satisfied_hardcons / self.num_constraints
        is_valid = ratio == 1.0
        # prints the ratio of satisfied constraints and prints if all constraints are satisfied
        logging.info(f"Ratio of satisfied constraints: {ratio}\nSuccess:{['no', 'yes'][int(is_valid)]}")
        return is_valid, round(time() * 1000 - start, 3)

    def evaluate(self, solution: dict) -> (float, float):
        """
        Calculates the quality of the solution.

        :param solution:
        :type solution: dict
        :return: Tour length, time it took to calculate the tour length
        :rtype: tuple(float, float)
        """
        start = time() * 1000

        logging.info("Checking the quality fo the solution:")
        # logging.info(solution)

        # count the number of satisfied clauses
        nr_satisfied_tests = len(*np.where([test.satisfied_by(solution) for test in self.application['tests']]))

        ratio_satisfied = nr_satisfied_tests / self.num_tests
        logging.info(f"Ratio of satisfied test clauses: {ratio_satisfied}.")

        return ratio_satisfied, round(time() * 1000 - start, 3)

    def save(self, path: str, iter_count: int) -> None:
        with open(f"{path}/constraints.cnf", 'w') as f_cons:
            dump(
                obj=self.application['constraints'], fp=f_cons,
                var_labels={str(literal): idx + 1 for idx, literal in enumerate(self.literals)}
            )
        with open(f"{path}/tests.cnf", 'w') as f_test:
            dump(
                obj=Or(self.application['tests']), fp=f_test,
                var_labels={str(literal): idx + 1 for idx, literal in enumerate(self.literals)}
            )
