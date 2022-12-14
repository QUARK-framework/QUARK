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
from nnf import And
from applications.Mapping import *
from solvers.Annealer import Annealer
from itertools import combinations
from time import time


class DinneenQubo(Mapping):
    """
    QUBO formulation for SAT as given by Dinneen -- see also the description in the QUARK paper (2202.03028).
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.solver_options = ["Annealer"]
        self.nr_vars = None

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

    def map(self, problem: (And, list), config: Config) -> (dict, float):
        """
        Performs the mapping into a QUBO formulation, as given by Dinneen. See also the QUARK paper.
        
        :param problem: 
        :type problem: any
        :param config: config with the parameters specified in Config class
        :type config: Config
        :return: dict with the QUBO, time it took to map it
        :rtype: tuple(dict, float)
        """""
        start = time() * 1000
        # extract hard and soft constraints from the generated problem
        hard, soft = problem
        # count the variables
        self.nr_vars = len(hard.vars().union(And(soft).vars()))
        lagrange = config['lagrange']
        # lagrange parameter is a factor of the number of soft constraints.
        lagrange *= len(soft)

        def _add_clause(curr_qubo_dict, clause, pos, weight):
            """
            Function that adds the QUBO terms corresponding to the clause and updates the QUBO dictionary
             accordingly. Additionally, the weight of the clause is taken into account.

            :param curr_qubo_dict:
            :param clause:
            :param pos:
            :param weight:
            :return:
            """

            def _check_and_add(dictionary, key, value):
                """
                Helper function that checks if key is present or not in dictionary and adds a value, adding the key
                if missing.

                :param dictionary:
                :param key:
                :param value:
                :return:
                """
                key = tuple(sorted(key))
                if key not in dictionary.keys():
                    dictionary[key] = value
                else:
                    dictionary[key] += value
                return dictionary

            cl_dict = {}
            for variable in clause.children:
                for variable_name in variable.vars():
                    # transforms the negations (0,1) into signs (-1, 1)
                    cl_dict[int(variable_name[1:])] = (int(variable.true) - 1 / 2) * 2

            # add the linear term of the auxiliary variable w
            curr_qubo_dict = _check_and_add(curr_qubo_dict, (pos, pos), 2 * weight)

            # add x linear terms and xw terms.
            for qvar, val in cl_dict.items():
                # qvar is the name of the var, val is the sign corresponding to whether the variable is negated or not.
                # linear x term:
                curr_qubo_dict = _check_and_add(curr_qubo_dict, (qvar, qvar), -weight * val)
                # x * w (aux. var.) term
                curr_qubo_dict = _check_and_add(curr_qubo_dict, (qvar, pos), -weight * val)
            # add combinations
            for q1, q2 in combinations(cl_dict.keys(), 2):
                curr_qubo_dict = _check_and_add(curr_qubo_dict, (q1, q2), weight * cl_dict[q1] * cl_dict[q2])

            return curr_qubo_dict

        qubo_dict = {}
        # first we add the hard constraints -- we add the lagrange parameter as weight
        for clause_ind, hard_clause in enumerate(hard):
            qubo_dict = _add_clause(qubo_dict, hard_clause, self.nr_vars + clause_ind, lagrange)
        # next, we add the soft constraints -- we start the enumeration at the final index corresponding to hard cons.
        for clause_ind, soft_clause in enumerate(soft):
            qubo_dict = _add_clause(qubo_dict, soft_clause, self.nr_vars + clause_ind + len(hard), 1)

        logging.info(f"Generate Dinneen QUBO with {self.nr_vars + len(hard) + len(soft)} binary variables."
                     f" Lagrange parameter used was: {config['lagrange']}.")
        return {"Q": qubo_dict}, round(time() * 1000 - start, 3)

    def reverse_map(self, solution: dict) -> (dict, float):
        """
        Reverse mapping of the solution obtained from the Dinneen QUBO.

        :param solution: dictionary containing the solution
        :type solution: dict
        :return: solution mapped accordingly, time it took to map it
        :rtype: tuple(dict, float)
        """
        start = time() * 1000
        mapped_sol = {}
        for i in range(self.nr_vars):
            # if variable not present in solution, its assignment does not matter
            if i not in solution.keys():
                mapped_sol[f'L{i}'] = True
            else:
                mapped_sol[f'L{i}'] = bool(solution[i])
        return mapped_sol, round(time() * 1000 - start, 3)

    def get_solver(self, solver_option: str) -> Annealer:

        if solver_option == "Annealer":
            return Annealer()
        else:
            raise NotImplementedError(f"Solver Option {solver_option} not implemented")
