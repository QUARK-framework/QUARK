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


class Application(ABC):
    """
    The application component defines the workload, comprising a dataset of increasing complexity, a validation, and an
    evaluation function.
    """

    def __init__(self, application_name):
        """
        Constructor method
        """
        self.application_name = application_name
        self.application = None
        self.mapping_options = []
        super().__init__()

    def get_application(self) -> any:
        """
        Getter that returns the application

        :return: self.application
        :rtype: any
        """
        return self.application

    @abstractmethod
    def get_solution_quality_unit(self) -> str:
        """
        Method to return the unit of the evaluation which is used to make the plots nicer.

        :return: String with the unit
        :rtype: str
        """

    @abstractmethod
    def get_parameter_options(self) -> dict:
        """
        Method to return the parameters needed to create a concrete problem of an application.

        Should always be in this format:

        .. code-block:: json

            {
               "parameter_name":{
                  "values":[1, 2, 3],
                  "description":"How many nodes do you need?"
               },
                "parameter_name_2":{
                  "values":["x", "y"],
                  "description":"Which type of problem do you want?"
               }
            }

        :return: Available application settings for this application
        :rtype: dict
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

    def process_solution(self, solution) -> (any, float):
        """
        Most of the time the solution has to be processed before it can be validated and evaluated
        This might not be necessary in all cases, so the default is to return the original solution.

        :param solution:
        :type solution: any
        :return: Processed solution and the execution time to process it
        :rtype: tuple(any, float)

        """
        return solution, 0

    @abstractmethod
    def validate(self, solution) -> (bool, float):
        """
        Check if the solution is a valid solution.

        :return: bool and the time it took to create it
        :param solution:
        :type solution: any
        :rtype: tuple(bool, float)

        """
        pass

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
    def save(self, path) -> None:
        """
        Function to save the concrete problem.

        :param path: path of the experiment directory for this run
        :type path: str
        :return:
        :rtype: None
        """
        pass

    @abstractmethod
    def get_mapping(self, mapping_option: str) -> any:
        """
        Return a mapping for an application.

        :param mapping_option: String with the option
        :rtype: str
        :return: instance of a mapping class
        :rtype: any
        """
        pass

    def get_available_mapping_options(self) -> list:
        """
        Get list of available mapping options.

        :return: list of mapping options
        :rtype: list
        """
        return self.mapping_options
