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

from modules.applications.Application import *


class Optimization(Application, ABC):
    """
    Optimization Module for QUARK, is used by all Optimization applications
    """

    @abstractmethod
    def validate(self, solution) -> (bool, float):
        """
        Check if the solution is a valid solution.

        :param solution: proposed solution
        :type solution: any
        :return: bool and the time it took to create it
        :rtype: tuple(bool, float)

        """
        pass

    @abstractmethod
    def get_solution_quality_unit(self) -> str:
        """
        Method to return the unit of the evaluation which is used to make the plots nicer.

        :return: String with the unit
        :rtype: str
        """

    @abstractmethod
    def evaluate(self, solution: any) -> (float, float):
        """
        Checks how good the solution is to allow comparison to other solutions.

        :param solution:
        :type solution: any
        :return: Evaluation and the time it took to create it
        :rtype: tuple(any, float)

        """
        pass

    @abstractmethod
    def generate_problem(self, config) -> any:
        """
        Depending on the config this method creates a concrete problem and returns it.

        :param config:
        :type config: dict
        :return:
        :rtype: any
        """
        pass

    def preprocess(self, input_data: any, config: dict, **kwargs) -> (any, float):
        """
        For optimization problems we generate the actual problem instance in the preprocess function.

        :param input_data: Usually not used for this method.
        :type input_data: any
        :param config: config for the problem creation.
        :type config:  dict
        :param kwargs: optional additional arguments.:
        :type kwargs:
        :return: tuple with output and the preprocessing time
        :rtype: (any, float)
        """
        start = time() * 1000
        output = self.generate_problem(config)
        return output, round(time() * 1000 - start, 3)

    def postprocess(self, input_data: any, config: dict, **kwargs) -> (any, float):
        """
        For optimization, we process the solution here, then validate and evaluate it.

        :param input_data:
        :type input_data: any
        :param config: config
        :type config: dict
        :param kwargs:
        :return: tuple with results and the postprocessing time
        :rtype: (any, float)
        """
        processed_solution = None
        try:
            processed_solution, time_to_process_solution = self.process_solution(
                input_data)
            solution_validity, time_to_validation = self.validate(
                processed_solution)
        except Exception as e:
            logging.exception(f"Exception on processing the solution: {e}")
            solution_validity = False
            time_to_process_solution = None
            time_to_validation = None
        if solution_validity and processed_solution:
            solution_quality, time_to_evaluation = self.evaluate(processed_solution)
        else:
            solution_quality = None
            time_to_evaluation = None

        self.metrics.add_metric_batch({"solution_validity": solution_validity, "solution_quality": solution_quality,
                                       "solution_quality_unit": self.get_solution_quality_unit(),
                                       "processed_solution": processed_solution})
        return solution_validity, sum(filter(None, [time_to_process_solution, time_to_validation, time_to_evaluation]))
