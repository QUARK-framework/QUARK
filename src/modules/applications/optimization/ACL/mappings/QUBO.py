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

import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import (QuadraticProgramToQubo, InequalityToEquality, IntegerToBinary,
                                            LinearEqualityToPenalty)

from modules.applications.Mapping import *
from utils import start_time_measurement, end_time_measurement


class Qubo(Mapping):
    """
    QUBO formulation for the ACL.

    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.submodule_options = ["Annealer"]
        self.global_variables = 0

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module

        :return: list of dict with requirements of this module
        :rtype: list[dict]
        """
        return [
            {
                "name": "numpy",
                "version": "1.26.4"
            },
            {
                "name": "qiskit-optimization",
                "version": "0.6.1"
            },
        ]

    def get_parameter_options(self):
        """
        Returns empty dict as this mapping has no configurable settings.

        :return: empty dictionary
        :rtype: dict
        """
        return {
        }

    class Config(TypedDict):
        """
        Empty config as this solver has no configurable settings.
        """
        pass

    def map_pulp_to_qiskit(self, problem: any):
        """
        Maps the problem dict to a quadratic program.

        :param problem: Problem formulation in dict form
        :type problem: dict
        :return: quadratic program in qiskit-optimization format
        :rtype: QuadraticProgram
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
        # Arguments:
        obj_arguments = {}
        for arg in problem["objective"]["coefficients"]:
            obj_arguments[arg["name"]] = arg["value"]

        # Maximize
        if problem["parameters"]["sense"] == -1:
            qp.maximize(linear=obj_arguments)
        # Minimize
        else:
            qp.minimize(linear=obj_arguments)

        # Constraints
        for constraint in problem["constraints"]:
            const_arguments = {}
            for arg in constraint["coefficients"]:
                const_arguments[arg["name"]] = arg["value"]
            sense = constraint["sense"]
            if sense == -1:
                const_sense = "LE"
            elif sense == 1:
                const_sense = "GE"
            else:
                const_sense = "E"
            qp.linear_constraint(linear=const_arguments, sense=const_sense, rhs=-1 * constraint["constant"],
                                 name=constraint["name"])
        return qp

    def convert_string_to_arguments(self, input_string: str):
        """
        Converts QUBO in string format to a list of separated arguments, used to construct the QUBO matrix.

        :param input_string: QUBO in raw string format
        :type input_string: str
        :return: list of arguments
        :rtype: list
        """
        terms = re.findall(r'[+\-]?[^+\-]+', input_string)
        # Convert the penalty string to a list of lists of the individual arguments in the penalty term
        result = [term.strip() for term in terms]
        separated_arguments = []
        first_item = True
        # Loop over all arguments in the penalty
        for argument in result:
            if first_item is True:
                # Remove "maximize" or minimize string from the first argument
                argument = argument[8:]
                first_item = False
            if "*" in argument:
                # The variables in each argument are connected by "*" signs. Here we split the variables
                elements = argument.split('*')
                # Convert string of numbers to floats
                new_argument = elements[0].strip()
                # Remove empty strings
                new_argument = [int(new_argument.replace(" ", "")) if new_argument.replace(" ", "").isdigit() else
                                float(new_argument.replace(" ", ""))]
                for el in elements[1:]:
                    new_argument += [el.strip()]
                separated_arguments.append(new_argument)
            else:
                separated_arguments.append(argument)
        return separated_arguments

    def construct_qubo(self, penalty: list[list], variables: list):
        """
        Creates QUBO matrix Q to solve linear problem of the form x^T * Q + x

        :param penalty: list of lists containing all non-zero elements of the QUBO matrix as strings
        :type penalty: list
        :param variables: listing of all variables used in the problem
        :type variables: list
        :return: QUBO in numpy array format
        :rtype: array
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
                    if type(argument) is list:
                        # squared variables in diagonals (x^2 == x)
                        if len(argument) == 2:
                            if any(isinstance(elem, str) and variable in elem for elem in argument) and col == row:
                                parameter += argument[0]
                        # Multiplication of different variables not on diagonal
                        if len(argument) == 3:
                            if variable in argument and variable2 in argument and variable > variable2:
                                parameter += argument[0]
                                # this value is already taking into account the factor 2 from quadratic term
                    # For the variables on the diagonal, if the parameter is zero, we still have to check the sign in
                    # front of the decision variable. If it is "-", we have to put "-1" on the diagonal.
                    elif type(argument) is str:
                        if variable in argument and variable2 in argument and variable == variable2:
                            if "-" in argument:
                                parameter += -1
                qubo[col, row] = parameter
        # Minimization problem
        qubo = -qubo.astype(int)

        return qubo

    def map(self, problem: any, config: Config) -> (dict, float):
        """
        Use Ising mapping of qiskit-optimize
        Converts linear program created with pulp to quadratic program to Ising with qiskit to QUBO matrix

        :param config: config with the parameters specified in Config class
        :type config: Config
        :return: dict with the QUBO, time it took to map it
        :rtype: tuple(dict, float)
        """
        start = start_time_measurement()

        # Map Linear problem from dictionary (generated by pulp) to quadratic program to QUBO
        qp = self.map_pulp_to_qiskit(problem)
        # print(qp.prettyprint())
        logging.info(qp.export_as_lp_string())
        ineq2eq = InequalityToEquality()
        qp_eq = ineq2eq.convert(qp)
        int2bin = IntegerToBinary()
        qp_eq_bin = int2bin.convert(qp_eq)
        lineq2penalty = LinearEqualityToPenalty(100)
        qubo = lineq2penalty.convert(qp_eq_bin)

        # get variables
        variables = []
        for variable in qubo.variables:
            variables.append(variable.name)

        # convert penalty term to string to QUBO
        qubo_string = str(qubo.objective)
        arguments = self.convert_string_to_arguments(qubo_string)
        qubo = self.construct_qubo(arguments, variables)

        self.global_variables = variables

        return {"Q": qubo}, end_time_measurement(start)

    def reverse_map(self, solution: dict) -> (dict, float):
        """
        Maps the solution back to the representation needed by the ACL class for validation/evaluation.

        :param solution: bit_string containing the solution
        :type solution: dict
        :return: solution mapped accordingly, time it took to map it
        :rtype: tuple(dict, float)
        """
        start = start_time_measurement()
        result = {"status": [0]}
        objective_value = 0
        variables = {}
        for bit in solution:
            if solution[bit] > 0:
                # We only care about assignments of vehicles to platforms:
                # We map the solution to the original variables
                if "x" in self.global_variables[bit]:
                    variables[self.global_variables[bit]] = solution[bit]
                    result["status"] = 'Optimal'  # TODO: I do not think every solution with at least one car is optimal
                    objective_value += solution[bit]
        result["variables"] = variables
        result["obj_value"] = objective_value
        return result, end_time_measurement(start)

    def get_default_submodule(self, option: str) -> Core:
        if option == "Annealer":
            from modules.solvers.Annealer import Annealer  # pylint: disable=C0415
            return Annealer()
        else:
            raise NotImplementedError(f"Solver Option {option} not implemented")
