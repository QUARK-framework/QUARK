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
from quark.utils import _get_instance_with_sub_options


class Mapping(ABC):
    """
    The task of the mapping module is to translate the application’s data and problem specification
    into a mathematical formulation suitable for a solver.
    """

    def __init__(self):
        """
        Constructor method.
        """
        self.solver_options = []
        self.sub_options = None
        super().__init__()

    @abstractmethod
    def map(self, problem: any, config: dict) -> tuple[any, float]:
        """
        Maps the given problem into a specific format a solver can work with. E.g. graph to QUBO.

        :param problem: Problem instance which should be mapped to the target representation
        :param config: Instance of class Config specifying the mapping settings
        :return: The mapped problem and the time it took to create the mapping
        """
        pass

    def reverse_map(self, solution: any) -> tuple[any, float]:
        """
        Maps the solution back to the original problem. This might not be necessary in all cases, so the default is
        to return the original solution. This might be needed to convert the solution to a representation needed
        for validation and evaluation.

        :param solution: Solution to be reversed back to its original representation
        :return: Mapped solution and the time it took to create it
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
        """
        pass

    def get_submodule(self, solver_option: str) -> any:
        """
        If self.sub_options is not None, a solver is instantiated according to the information given in sub_options.
        Otherwise, get_solver is called as fall back.

        :param solver_option: The option for the solver
        :return: Instance of a solver class
        """
        if self.sub_options is None:
            return self.get_solver(solver_option)
        else:
            return _get_instance_with_sub_options(self.sub_options, solver_option)

    @abstractmethod
    def get_solver(self, solver_option: str) -> any:
        """
        Returns the default solver for a given string. This applies only if
        self.sub_options is None. See get_submodule.

        :param solver_option: desired solver option
        :return: Instance of solver class
        """
        pass

    def get_available_solver_options(self) -> list:
        """
        Returns all available solvers.

        :return: List of solvers
        """
        if self.sub_options is None:
            return self.solver_options
        else:
            return [option["name"] for option in self.sub_options]
