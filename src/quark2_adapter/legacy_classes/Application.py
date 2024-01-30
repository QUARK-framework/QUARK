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
from typing import final

from utils import _get_instance_with_sub_options


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
        self.sub_options = []

        self.problem = None
        self.problems = {}
        self.conf_idx = None

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

    def regenerate_on_iteration(self, config: dict) -> bool:
        """Overwrite this to return True if the problem should be newly generated
        on every iteration. Typically, this will be the case if the problem is taken
        from a statistical ensemble e.g. an erdos-renyi graph.
        :param config: the application configuration
        :type config: dict
        :return: whether the problem should be recreated on every iteration. Returns False if not overwritten.
        :rtype: bool
        """
        return False

    @final
    def init_problem(self, config, conf_idx: int, iter_count: int, path):
        """
        This method is called on every iteration and calls generate_problem if necessary.
        conf_idx identifies the application configuration.
        Note that when there are several mappings, solvers or devices this method is
        called several times with the same conf_idx and rep_count. In this case
        the problem will be the same if conf_idx and rep_count are both the same.

        The implementation assumes that the loop over the rep_count is within the loop
        over the different application configurations (conf_idx).

        :param config: the application configuration
        :type config: dict
        :param conf_idx: the index of the application configuration
        :conf_idx: int
        :param iter_count: the repetition count (starting with 1)
        :type iter_count: int
        :param path: the path used to save each newly generated problem instance
        :type path: str
        :return: the current problem instance
        :rtype: any
        """

        if conf_idx != self.conf_idx:
            self.problems = {}
            self.conf_idx = conf_idx

        key = iter_count if self.regenerate_on_iteration(config) else "dummy"
        if key in self.problems:
            self.problem = self.problems[key]
        else:
            self.problem = self.generate_problem(config, iter_count)
            self.problems[key] = self.problem
            self.save(path, iter_count)
        return self.problem

    @abstractmethod
    def generate_problem(self, config: dict, iter_count: int) -> any:
        """
        Depending on the config this method creates a concrete problem and returns it.

        :param config:
        :type config: dict
        :param iter_count: the iteration count
        :type iter_count: int
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
    def save(self, path: str, iter_count: int) -> None:
        """
        Function to save the concrete problem.

        :param path: path of the experiment directory for this run
        :type path: str
        :param iter_count: the iteration count
        :type iter_count: int
        :return:
        :rtype: None
        """
        pass

    def get_submodule(self, mapping_option: str) -> any:
        """
        If self.sub_options is not None, a mapping is instantiated according to the information given in
        self.sub_options. Otherwise, get_mapping is called as fall back.

        :param mapping_option: String with the option
        :type mapping_option: str
        :return: instance of a mapping class
        :rtype: any
        """
        if self.sub_options is None:
            return self.get_mapping(mapping_option)
        else:
            return _get_instance_with_sub_options(self.sub_options, mapping_option)

    @abstractmethod
    def get_mapping(self, mapping_option: str) -> any:
        """
        Return a default mapping for an application. This applies only if
        self.sub_options is None. See get_submodule.

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
        if self.sub_options is None:
            return self.mapping_options
        else:
            return [o["name"] for o in self.sub_options]
