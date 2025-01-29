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

from itertools import combinations
from typing import TypedDict
import logging

from nnf import And

from quark.modules.applications.Mapping import Mapping, Core
from quark.utils import start_time_measurement, end_time_measurement


class DinneenQUBO(Mapping):
    """
    QUBO formulation for SAT as given by Dinneen -- see also the description in the QUARK paper (2202.03028).
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__()
        self.submodule_options = ["Annealer"]
        self.nr_vars = None

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module.

        :return: List of dict with requirements of this module
        """
        return [{"name": "nnf", "version": "0.4.1"}]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this mapping.

        :return: Dictionary with parameter options
        .. code-block:: python

            return {
                    "lagrange": {
                        "values": [0.1, 1, 2],
                        "description": "What Lagrange param. to multiply with the number of (hard) constr.?"
                    }
                }
        """
        return {
            "lagrange": {
                "values": [0.1, 1, 2],
                "description": "What Lagrange parameter to multiply with the number of (hard) constraints?"
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config.

        .. code-block:: python

             lagrange: float

        """
        lagrange: float

    def map(self, problem: tuple[And, list], config: Config) -> tuple[dict, float]:
        """
        Performs the mapping into a QUBO formulation, as given by Dinneen. See also the QUARK paper.

        :param problem: SAT problem
        :param config: Config with the parameters specified in Config class
        :return: Tuple with the QUBO, time it took to map it
        """""
        start = start_time_measurement()

        # Extract hard and soft constraints from the generated problem
        hard, soft = problem

        # Count the variables
        self.nr_vars = len(hard.vars().union(And(soft).vars()))
        lagrange = config['lagrange']
        # Lagrange parameter is a factor of the number of soft constraints.
        lagrange *= len(soft)

        def _add_clause(curr_qubo_dict: dict[tuple[int, int], float],
                        clause: any,
                        pos: int,
                        weight: float) -> dict[tuple[int, int], float]:
            """
            Function that adds the QUBO terms corresponding to the clause and updates the QUBO dictionary
             accordingly. Additionally, the weight of the clause is taken into account.

            :param curr_qubo_dict: Current QUBO dictionary
            :param clause: Clause to be added
            :param pos: Position of the auxiliary variable
            :param weight: Weight of the clause
            :return: Updated QUBO dictionary
            """

            def _check_and_add(dictionary: dict, key: tuple[int, int], value: float) -> dict:
                """
                Helper function that checks if key is present or not in dictionary and adds a value, adding the key
                if missing.

                :param dictionary: Dictionary to be updated
                :param key: Key to check in the dictionary
                :param value: Value to add to the key
                :return: Updated dictionary
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
                    # Transforms the negations (0,1) into signs (-1, 1)
                    cl_dict[int(variable_name[1:])] = (int(variable.true) - 1 / 2) * 2

            # Add the linear term of the auxiliary variable w
            curr_qubo_dict = _check_and_add(curr_qubo_dict, (pos, pos), 2 * weight)

            # Add x linear terms and xw terms.
            for qvar, val in cl_dict.items():
                # qvar is the name of the var, val is the sign corresponding to whether the variable is negated or not.
                # linear x term:
                curr_qubo_dict = _check_and_add(curr_qubo_dict, (qvar, qvar), -weight * val)
                # x * w (aux. var.) term
                curr_qubo_dict = _check_and_add(curr_qubo_dict, (qvar, pos), -weight * val)
            # Add combinations
            for q1, q2 in combinations(cl_dict.keys(), 2):
                curr_qubo_dict = _check_and_add(curr_qubo_dict, (q1, q2), weight * cl_dict[q1] * cl_dict[q2])

            return curr_qubo_dict

        qubo_dict = {}
        # Add the hard constraints and add the lagrange parameter as weight
        for clause_ind, hard_clause in enumerate(hard):
            qubo_dict = _add_clause(qubo_dict, hard_clause, self.nr_vars + clause_ind, lagrange)

        # Add the soft constraints and start the enumeration at the final index corresponding to hard cons.
        for clause_ind, soft_clause in enumerate(soft):
            qubo_dict = _add_clause(qubo_dict, soft_clause, self.nr_vars + clause_ind + len(hard), 1)

        logging.info(
            f"Generate Dinneen QUBO with {self.nr_vars + len(hard) + len(soft)} binary variables."
            f" Lagrange parameter used was: {config['lagrange']}."
        )

        return {"Q": qubo_dict}, end_time_measurement(start)

    def reverse_map(self, solution: dict) -> tuple[dict, float]:
        """
        Reverse mapping of the solution obtained from the Dinneen QUBO.

        :param solution: Dictionary containing the solution
        :return: Solution mapped accordingly, time it took to map it
        """
        start = start_time_measurement()
        mapped_sol = {}
        for i in range(self.nr_vars):
            # if variable not present in solution, its assignment does not matter
            if i not in solution.keys():
                mapped_sol[f'L{i}'] = True
            else:
                mapped_sol[f'L{i}'] = bool(solution[i])

        return mapped_sol, end_time_measurement(start)

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
