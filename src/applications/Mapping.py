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

from abc import ABC, abstractmethod
from time import time


class Mapping(ABC):
    """
    The task of the mapping module is to translate the application’s data and problem specification into a mathematical
    formulation suitable for a solver.
    """

    def __init__(self):
        """
        Constructor method
        """
        self.solver_options = []
        super().__init__()

    @abstractmethod
    def map(self, problem, config) -> (any, float):
        """
        Maps the given problem into a specific format a solver can work with. E.g. graph to QUBO.

        :param config: instance of class Config specifying the mapping settings
        :param problem: problem instance which should be mapped to the target representation
        :return: Must always return the mapped problem and the time it took to create the mapping
        :rtype: tuple(any, float)
        """
        pass

    def reverse_map(self, solution) -> (any, float):
        """
        Maps the solution back to the original problem. This might not be necessary in all cases, so the default is
        to return the original solution. This might be needed to convert the solution to a representation needed
        for validation and evaluation.

        :param solution:
        :type solution: any
        :return: Mapped solution and the time it took to create it
        :rtype: tuple(any, float)

        """
        return solution, 0

    @abstractmethod
    def get_parameter_options(self) -> dict:
        """
        Method to return the parameters to fine tune the mapping.

        Should always be in this format:

        .. code-block:: json

            {
               "parameter_name":{
                  "values":[1, 2, 3],
                  "description":"How to scale your Lagrangian?"
               }
            }

        :return: Returns the available parameter options of this mapping
        :rtype: dict
        """
        pass

    @abstractmethod
    def get_solver(self, solver_option: str) -> any:
        """
        Returns the solver for a given string.

        :param solver_option: desired solver
        :type solver_option: str
        :return: instance of solver class
        :rtype: any
        """
        pass

    def get_available_solver_options(self) -> list:
        """
        Returns all available solvers.

        :return: list of solvers
        :rtype: list
        """
        return self.solver_options
