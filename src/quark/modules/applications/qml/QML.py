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

from quark.modules.applications.Application import Application


class QML(Application, ABC):
    """
    qml Module for QUARK, is used by all qml applications.
    """

    @abstractmethod
    def generate_problem(self, config: dict) -> any:
        """
        Creates a concrete problem and returns it.

        :param config: Configuration dictionary
        :return: Generated problem
        """
        pass

    def save(self, path: str, iter_count: int) -> None:
        """
        Placeholder method for saving output to a file.

        :param path: Path to save the file
        :param iter_count: Iteration count
        """
        pass
