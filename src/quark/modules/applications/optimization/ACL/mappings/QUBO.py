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
import re
import logging

import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import (
    InequalityToEquality, IntegerToBinary,
    LinearEqualityToPenalty
)

from quark.modules.applications.Mapping import Mapping, Core
from quark.utils import start_time_measurement, end_time_measurement

# TODO Large chunks of this code is duplicated in ACL.mappings.ISING -> unify


class Qubo(Mapping):
    """
    QUBO formulation for the ACL.
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__()
        self.submodule_options = ["Annealer"]
        self.global_variables = []

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module.

        :return: list of dict with requirements of this module
        """
        return [
            {"name": "numpy", "version": "1.26.4"},
            {"name": "qiskit-optimization", "version": "0.6.1"},
        ]

    def get_parameter_options(self) -> dict:
        """
        Returns empty dict as this mapping has no configurable settings.

        :return: Empty dictionary
        """
        return {}

    class Config(TypedDict):
        """
        Empty config as this solver has no configurable settings.
        """
        pass

    def map_pulp_to_qiskit(self, problem: dict) -> QuadraticProgram:
        """
        Maps the problem dict to a quadratic program.

        :param problem: Problem formulation in dict form
        :return: Quadratic program in qiskit-optimization format
        """
        # Details at:
        # https://coin-or.github.io/pulp/guides/how_to_export_models.html
        # https://qiskit.org/documentation/stable/0.26/tutorials/optimization/2_converters_for_quadratic_programs.html

        qp = QuadraticProgram()

        # Variables
        for variable_dict in problem["variables"]:
            if variable_dict["cat"] == "Integer":
                lb = variable_dict["lowBound"]
                ub = variable_dict["upBound"]
                name = variable_dict["name"]
                # If the integer variable is actually a binary variable
                if lb == 0 and ub == 1:
                    qp.binary_var(name)
                # If the integer variable is non-binary
                else:
                    qp.integer_var(lowerbound=lb, upperbound=ub, name=name)

        # Objective function
        obj_arguments = {arg["name"]: arg["value"] for arg in problem["objective"]["coefficients"]}

        # Maximize
        if problem["parameters"]["sense"] == -1:
            qp.maximize(linear=obj_arguments)
        # Minimize
        else:
            qp.minimize(linear=obj_arguments)

        # Constraints
        for constraint in problem["constraints"]:
            const_arguments = {arg["name"]: arg["value"] for arg in constraint["coefficients"]}
            sense = constraint["sense"]
            const_sense = "LE" if sense == -1 else "GE" if sense == 1 else "E"

            qp.linear_constraint(
                linear=const_arguments,
                sense=const_sense,
                rhs=-1 * constraint["constant"],
                name=constraint["name"]
            )

        return qp

    def convert_string_to_arguments(self, input_string: str) -> list[any]:
        """
        Converts QUBO in string format to a list of separated arguments,
        used to construct the QUBO matrix.

        :param input_string: QUBO in raw string format
        :return: List of arguments
        """
        terms = re.findall(r'[+\-]?[^+\-]+', input_string)
        # Convert the penalty string to a list of lists of the individual arguments in the penalty term
        result = [term.strip() for term in terms]
        separated_arguments = []
        first_item = True

        for argument in result:
            if first_item:
                # Remove "maximize" or minimize string from the first argument
                argument = argument[8:]
                first_item = False
            if "*" in argument:
                # The variables in each argument are connected by "*" signs. Here we split the variables
                elements = argument.split('*')
                # Convert string of numbers to floats
                new_argument = elements[0].strip()
                # Remove empty strings
                new_argument = [int(new_argument.replace(" ", "")) if new_argument.replace(" ", "").isdigit()
                                else float(new_argument.replace(" ", ""))]
                new_argument += [el.strip() for el in elements[1:]]
                separated_arguments.append(new_argument)
            else:
                separated_arguments.append(argument)

        return separated_arguments

    def construct_qubo(self, penalty: list[list], variables: list[str]) -> np.ndarray:
        """
        Creates QUBO matrix Q to solve linear problem of the form x^T * Q + x.

        :param penalty: List of lists containing all non-zero elements of the QUBO matrix as strings
        :param variables: Listing of all variables used in the problem
        :return: QUBO in numpy array format
        """
        # Create empty qubo matrix
        count_variables = len(variables)
        qubo = np.zeros((count_variables, count_variables))

        # Iterate through all the variables twice (x^T, x)
        for col, variable in enumerate(variables):
            for row, variable2 in enumerate(variables):
                # Save the parameters (values in the qubo)
                parameter = 0
                for argument in penalty:
                    if isinstance(argument, list):
                        # squared variables in diagonals (x^2 == x)
                        if (
                            len(argument) == 2
                            and any(isinstance(elem, str) and variable in elem for elem in argument)
                            and col == row
                        ):
                            parameter += argument[0]
                        # Multiplication of different variables not on diagonal
                        if (
                            len(argument) == 3
                            and variable in argument and variable2 in argument and variable > variable2
                        ):
                            parameter += argument[0]
                            # This value is already taking into account the factor 2 from quadratic term
                            # For the variables on the diagonal, if the parameter is zero
                            # We still have to check the sign in
                            # front of the decision variable. If it is "-", we have to put "-1" on the diagonal.
                    elif (isinstance(argument, str) and variable in argument
                          and variable2 in argument and variable == variable2):
                        if "-" in argument:
                            parameter += -1

                qubo[col, row] = parameter

        # Minimization problem
        qubo = -qubo.astype(int)

        return qubo

    def map(self, problem: dict, config: Config) -> tuple[dict, float]:
        """
        Converts linear program created with pulp to quadratic program to Ising with qiskit to QUBO matrix.

        :param problem: Dict containing the problem parameters
        :param config: Config with the parameters specified in Config class
        :return: Dict with the QUBO, time it took to map it
        """
        start = start_time_measurement()

        # Map Linear problem from dictionary (generated by pulp) to quadratic program to QUBO
        qp = self.map_pulp_to_qiskit(problem)
        logging.info(qp.export_as_lp_string())

        ineq2eq = InequalityToEquality()
        qp_eq = ineq2eq.convert(qp)

        int2bin = IntegerToBinary()
        qp_eq_bin = int2bin.convert(qp_eq)

        lineq2penalty = LinearEqualityToPenalty(100)
        qubo = lineq2penalty.convert(qp_eq_bin)

        variables = [variable.name for variable in qubo.variables]

        # convert penalty term to string to QUBO
        qubo_string = str(qubo.objective)
        arguments = self.convert_string_to_arguments(qubo_string)
        qubo_matrix = self.construct_qubo(arguments, variables)

        self.global_variables = variables

        return {"Q": qubo_matrix}, end_time_measurement(start)

    def reverse_map(self, solution: dict) -> tuple[dict, float]:
        """
        Maps the solution back to the representation needed by the ACL class for validation/evaluation.

        :param solution: bit_string containing the solution
        :return: Solution mapped accordingly, time it took to map it
        """
        start = start_time_measurement()

        result = {"status": [0]}
        objective_value = 0
        variables = {}
        for bit in solution:
            if solution[bit] > 0 and "x" in self.global_variables[bit]:
                # We only care about assignments of vehicles to platforms:
                # We map the solution to the original variables
                variables[self.global_variables[bit]] = solution[bit]
                result["status"] = 'Optimal'  # TODO: I do not think every solution with at least one car is optimal
                objective_value += solution[bit]

        result["variables"] = variables
        result["obj_value"] = objective_value

        return result, end_time_measurement(start)

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
