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

from braket.devices import LocalSimulator as LocalSimulatorBraket

from quark.modules.devices.braket.Braket import Braket


class LocalSimulator(Braket):
    """
    Class for using the local Amazon Braket simulator.
    """

    def __init__(self, device_name: str):
        """
        Constructor method for initializing the LocalSimulator class.

        :param device_name: Name of the device.
        """
        super().__init__(device_name=device_name)
        self.device = LocalSimulatorBraket()
        self.submodule_options = []

    def get_parameter_options(self) -> dict:
        """
        Returns empty dictionary as this solver has no configurable settings.

        :return: An empty dictionary
        """
        return {}

    def get_default_submodule(self, option: str) -> None:
        """
        Raises ValueError as this module has no submodules.

        :param option: Option name
        :raises ValueError: If called, since this module has no submodules
        """
        raise ValueError("This module has no submodules.")
