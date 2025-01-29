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
from quark.modules.Core import Core


class Mapping(Core, ABC):
    """
    This module translates the input data and problem specification from the parent module,
    e.g., the application into a mathematical formulation suitable the submodule, e.g., a solver.
    """

    def preprocess(self, input_data: any, config: dict, **kwargs) -> tuple[any, float]:
        """
        Maps the data to the correct target format.

        :param input_data: Data which should be mapped
        :param config: Config of the mapping
        :param kwargs: Optional keyword arguments
        :return: Tuple with mapped problem and the time it took to map it
        """
        output, preprocessing_time = self.map(input_data, config)
        return output, preprocessing_time

    def postprocess(self, input_data: any, config: dict, **kwargs) -> tuple[any, float]:
        """
        Reverse transformation/mapping from the submodule's format to the mathematical formulation
        suitable for the parent module.

        :param input_data: Data which should be reverse-mapped
        :param config: Config of the reverse mapping
        :param kwargs: Optional keyword arguments
        :return: Tuple with reverse-mapped problem and the time it took to map it
        """
        output, postprocessing_time = self.reverse_map(input_data)
        return output, postprocessing_time

    @abstractmethod
    def map(self, problem: any, config: dict) -> tuple[any, float]:
        """
        Maps the given problem into a specific format suitable for the submodule, e.g., a solver.

        :param config: Instance of class Config specifying the mapping settings
        :param problem: Problem instance which should be mapped to the target representation
        :return: Mapped problem and the time it took to map it
        """
        pass

    def reverse_map(self, solution: any) -> tuple[any, float]:
        """
        Maps the solution back to the original problem. This might not be necessary in all cases, so the default is
        to return the original solution. This might be needed to convert the solution to a representation needed
        for validation and evaluation.

        :param solution: Solution provided by submodule, e.g., the Solver class
        :return: Reverse-mapped solution and the time it took to create it
        """
        return solution, 0
