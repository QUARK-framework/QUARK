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


class Application(Core, ABC):
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

        super().__init__()

    def get_application(self) -> any:
        """
        Getter that returns the application

        :return: self.application
        :rtype: any
        """
        return self.application

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
