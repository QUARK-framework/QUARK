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

import dwave.samplers

from modules.devices.Device import Device
from modules.Core import Core


class SimulatedAnnealingSampler(Device):
    """
    Class for D-Waves neal simulated annealer
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__(device_name="simulated annealer")
        self.device = dwave.samplers.SimulatedAnnealingSampler()
        self.submodule_options = []

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module

        :return: list of dict with requirements of this module
        :rtype: list[dict]
        """
        return [
            {
                "name": "dwave-samplers",
                "version": "1.3.0"
            }
        ]

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
