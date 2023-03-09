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
    The task of the mapping module is to translate the application’s data and problem specification into a mathematical
    formulation suitable for a solver.
    """

    def preprocess(self, input_data: any, config: dict, **kwargs) -> (any, float):
        """
        In this module, the preprocessing step is mapping the data to the correct target format.
        :param input_data:
        :type input_data: any
        :param config:
        :type config: dict
        :param kwargs:
        :type kwargs: dict
        :return: tuple with mapped problem and the time it took to map it
        :rtype: (any, float)
        """
        output, preprocessing_time = self.map(input_data, config)
        return output, preprocessing_time

    def postprocess(self, input_data: any, config: dict, **kwargs) -> (any, float):
        """
        Does the reverse transformation/mapping

        :param input_data:
        :type input_data: any
        :param config:
        :type config: dict
        :param kwargs:
        :type kwargs: dict
        :return:
        """
        output, postprocessing_time = self.reverse_map(input_data)
        return output, postprocessing_time

    @abstractmethod
    def map(self, problem: any, config: dict) -> (any, float):
        """
        Maps the given problem into a specific format a solver can work with. E.g. graph to QUBO.

        :param config: instance of class Config specifying the mapping settings
        :type config: dict
        :param problem: problem instance which should be mapped to the target representation
        :type problem: any
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
