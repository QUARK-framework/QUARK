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

from quark.modules.devices.Device import Device


class Pulser(Device, ABC):
    """
    Abstract class to use the Pulser devices.
    """

    def __init__(self, device_name: str):
        """
        Constructor method.

        :param device_name: Name of the Pulser device.
        """
        super().__init__(device_name)
        self.device = None
        self.backend = None

    def get_backend(self) -> any:
        """
        Returns backend.

        :return: Instance of the backend class
        """
        return self.backend

    @abstractmethod
    def get_backend_config(self) -> any:
        """
        Returns backend configurations.

        :return: Instance of the backend config class
        """
        pass

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module.

        :return: List of dict with requirements of this module
        """
        return [{"name": "pulser", "version": "1.1.1"}]
