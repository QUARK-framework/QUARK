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


class Solver(Core, ABC):
    """
    The solver is responsible for finding feasible and high-quality solutions of the formulated problem, i.e., of the
    defined objective function.
    """

    def postprocess(self, input_data: any, config: dict, **kwargs) -> tuple[any, float]:
        """
        The actual solving process is done here, using the device which is provided by the device submodule
        and the problem data provided by the parent module.

        :param input_data: Data passed to the run function of the solver
        :param config: Solver config
        :param kwargs: Optional keyword arguments
        :return: Output and time needed
        """
        output, elapsed_time, additional_metrics = self.run(self.preprocessed_input, input_data, config, **kwargs)
        self.metrics.add_metric_batch(additional_metrics)
        return output, elapsed_time

    @abstractmethod
    def run(self, mapped_problem: any, device_wrapper: any, config: any, **kwargs) -> tuple[any, float, dict]:
        """
        This function runs the solving algorithm on a mapped problem instance and returns a solution.

        :param mapped_problem: A representation of the problem that the solver can solve
        :param device_wrapper: A device the solver can leverage for the algorithm
        :param config: Settings for the solver such as hyperparameters
        :param kwargs: Optional additional settings
        :return: Solution, the time it took to compute it and some optional additional information
        """
        pass
