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

from modules.devices.braket.Braket import Braket
from modules.Core import Core


class LocalSimulator(Braket):
    """
    Class for using the local Amazon Braket simulator
    """

    def __init__(self, device_name: str):
        """
        Constructor method
        """
        super().__init__(device_name=device_name)
        self.device = LocalSimulatorBraket()
        self.submodule_options = []

    def get_parameter_options(self) -> dict:
        """
        Returns empty dict as this solver has no configurable settings

        :return: empty dict
        :rtype: dict
        """
        return {

        }

    def get_default_submodule(self, option: str) -> Core:
        raise ValueError("This module has no submodules.")
