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
import logging

from quark.modules.applications.Application import Application
from quark.utils import start_time_measurement, end_time_measurement


class Optimization(Application, ABC):
    """
    Optimization Module for QUARK, is used by all Optimization applications.
    """

    @abstractmethod
    def validate(self, solution: any) -> tuple[bool, float]:
        """
        Checks if the solution is a valid solution.

        :param solution: Proposed solution
        :return: Bool value if solution is valid and the time it took to validate the solution
        """
        pass

    @abstractmethod
    def get_solution_quality_unit(self) -> str:
        """
        Returns the unit of the evaluation.

        :return: String with the unit
        """
        pass

    @abstractmethod
    def evaluate(self, solution: any) -> tuple[float, float]:
        """
        Checks how good the solution is.

        :param solution: Provided solution
        :return: Tuple with the evaluation and the time it took to create it
        """
        pass

    @abstractmethod
    def generate_problem(self, config: dict) -> any:
        """
        Creates a concrete problem and returns it.

        :param config: Configuration for problem creation
        :return: Generated problem
        """
        pass

    def process_solution(self, solution: any) -> tuple[any, float]:
        """
        Most of the time the solution has to be processed before it can be validated and evaluated.
        This might not be necessary in all cases, so the default is to return the original solution.

        :param solution: Proposed solution
        :return: Tuple with processed solution and the execution time to process it
        """
        return solution, 0.0

    def preprocess(self, input_data: any, config: dict, **kwargs) -> tuple[any, float]:
        """
        For optimization problems, we generate the actual problem instance in the preprocess function.

        :param input_data: Input data (usually not used in this method)
        :param config: Config for the problem creation
        :param kwargs: Optional additional arguments
        :return: Tuple with output and the preprocessing time
        """
        start = start_time_measurement()
        output = self.generate_problem(config)
        return output, end_time_measurement(start)

    def postprocess(self, input_data: any, config: dict, **kwargs) -> tuple[any, float]:
        """
        For optimization problems, we process the solution here, then validate and evaluate it.

        :param input_data: Data which should be evaluated for this optimization problem
        :param config: Config for the problem creation
        :param kwargs: Optional additional arguments
        :return: Tuple with results and the postprocessing time
        """
        processed_solution = None
        try:
            processed_solution, time_to_process_solution = self.process_solution(input_data)
            solution_validity, time_to_validation = self.validate(processed_solution)
        except Exception as e:
            logging.exception(f"Exception on processing the solution: {e}")
            solution_validity = False
            time_to_process_solution = None
            time_to_validation = None

        if solution_validity and (processed_solution is not None):
            solution_quality, time_to_evaluation = self.evaluate(processed_solution)
            self.visualize_solution(processed_solution, f"{kwargs["store_dir"]}/solution.pdf")
        else:
            solution_quality = None
            time_to_evaluation = None

        self.metrics.add_metric_batch({
            "application_score_value": solution_quality,
            "application_score_unit": self.get_solution_quality_unit(),
            "application_score_type": str(float),
            "processed_solution": processed_solution,
            "time_to_process_solution": time_to_process_solution,
            "time_to_validation": time_to_validation,
            "time_to_evaluation": time_to_evaluation
        })

        return solution_validity, sum(filter(None, [
            time_to_process_solution, time_to_validation, time_to_evaluation
        ]))

    def visualize_solution(self, processed_solution: any, path: str) -> None:
        """
        Creates visualizations of a processed and validated solution and writes them to disk.
        Override if applicable. Default is to do nothing.

        :param processed_solution: A solution that was already processed by :func:`process_solution`
        :param path: File path for the plot
        :returns: None
        """
        pass
