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


class Application(Core, ABC):
    """
    The application component defines the workload, comprising a dataset of increasing complexity, a validation, and an
    evaluation function.
    """

    def __init__(self, application_name: str):
        """
        Constructor method.
        """
        super().__init__(application_name)
        self.application_name = self.name
        self.application = None

    def get_application(self) -> any:
        """
        Gets the application.

        :return: self.application
        """
        return self.application

    @abstractmethod
    def save(self, path: str, iter_count: int) -> None:
        """
        Saves the concrete problem.

        :param path: Path of the experiment directory for this run
        :param iter_count: The iteration count
        """
        pass
