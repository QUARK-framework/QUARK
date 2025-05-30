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

from dwave.system import DWaveSampler, AutoEmbeddingComposite

from modules.devices.device import Device
from modules.core import Core


class QuantumAnnealingSampler(Device):
    """
    Class for D-Wave quantum annealer.
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__(device_name="quantum annealer dwave")

        dwave_token = input("What is your Dwave-Solver-API-Token? \n Copy-paste it here from your DWave-Leap account: ")
        self.device = AutoEmbeddingComposite(DWaveSampler(token=dwave_token))
        self.submodule_options = []

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module.

        :return: List of dict with requirements of this module
        """
        return [{"name": "dwave-system", "version": "1.23.0"}]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this mapping.

        :return: An empty dict
        """
        return {

        }

    def get_default_submodule(self, option: str) -> Core:
        """
        This module has no submodules.

        :param option: Submodule option (not applicable)
        :raises ValueError: Always raised because this module has no submodules
        """
        raise ValueError("This module has no submodules.")
