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

from modules.Core import *


class Mapping(Core, ABC):
    """
    This module translates the input data and problem specification from the parent module, e.g.,
    the application into a mathematical formulation suitable the submodule, e.g., a solver.
    """

    def preprocess(self, input_data: any, config: dict, **kwargs) -> (any, float):
        """
        Maps the data to the correct target format

        :param input_data: Data which should be mapped
        :type input_data: any
        :param config: Config of the mapping
        :type config: dict
        :param kwargs: Optional keyword arguments
        :type kwargs: dict
        :return: Tuple with mapped problem and the time it took to map it
        :rtype: (any, float)
        """
        output, preprocessing_time = self.map(input_data, config)
        return output, preprocessing_time

    def postprocess(self, input_data: any, config: dict, **kwargs) -> (any, float):
        """
        Reverse transformation/mapping from the submodule's format to the mathematical formulation
        suitable for the parent module

        :param input_data: Data which should be reverse-mapped
        :type input_data: any
        :param config: Config of the reverse mapping
        :type config: dict
        :param kwargs: optional keyword arguments
        :type kwargs: dict
        :return: Tuple with reverse-mapped problem and the time it took to map it
        :rtype: (any,float)
        """
        output, postprocessing_time = self.reverse_map(input_data)
        return output, postprocessing_time

    @abstractmethod
    def map(self, problem: any, config: dict) -> (any, float):
        """
        Maps the given problem into a specific format suitable for the submodule, e.g., a solver

        :param config: Instance of class Config specifying the mapping settings
        :type config: dict
        :param problem: Problem instance which should be mapped to the target representation
        :type problem: any
        :return: Mapped problem and the time it took to map it
        :rtype: tuple(any, float)
        """
        pass

    def reverse_map(self, solution) -> (any, float):
        """
        Maps the solution back to the original problem. This might not be necessary in all cases, so the default is
        to return the original solution. This might be needed to convert the solution to a representation needed
        for validation and evaluation.

        :param solution: Solution provided by submodule, e.g., the Solver class
        :type solution: any
        :return: Reverse-mapped solution and the time it took to create it
        :rtype: tuple(any, float)

        """
        return solution, 0
