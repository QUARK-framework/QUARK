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

from typing import TypedDict

import pulser
from pulser.devices import MockDevice
from pulser_simulation import QutipBackend
from pulser.noise_model import NoiseModel
from pulser.backend.config import EmulatorConfig

from quark.modules.devices.pulser.Pulser import Pulser


class MockNeutralAtomDevice(Pulser):
    """
    Class for using the local mock Pulser simulator for neutral atom devices.
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__(device_name="mock neutral atom device")
        self.device = MockDevice
        self.backend = QutipBackend
        self.submodule_options = []

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this application.

        :return: Configurable settings for the mock neutral atom device
        """
        return {
            "doppler": {
                "values": [False, True],
                "description": "Simulate doppler noise? Has a large impact on performance!"
            },
            "amplitude": {
                "values": [False, True],
                "description": "Simulate amplitude noise? Has a large impact on performance!"
            },
            "SPAM": {
                "values": [False, True],
                "description": "Simulate SPAM noise? Has a large impact on performance!"
            },
            "dephasing": {
                "values": [False, True],
                "description": "Simulate dephasing noise? Has a large impact on performance!"
            },
        }

    class Config(TypedDict):
        """
        Attributes of a valid config.
        """
        doppler: bool
        amplitude: bool
        SPAM: bool
        dephasing: bool

    def get_backend_config(self) -> pulser.backend.config.EmulatorConfig:
        """
        Returns backend configurations.

        :return: Backend config for the emulator
        """
        noise_params = {}
        if "relaxation_rate" in self.config:
            noise_params["relaxation_rate"] = self.config["relaxation_rate"]
        if "dephasing_rate" in self.config:
            noise_params["dephasing_rate"] = self.config["dephasing_rate"]
        if "depolarizing_rate" in self.config:
            noise_params["depolarizing_rate"] = self.config["depolarizing_rate"]

        # Log the noise parameters for debugging
        print("Configured noise parameters:", noise_params)

        # Create the NoiseModel using parameters from self.config
        noise_model = NoiseModel(**noise_params)

        # Return the EmulatorConfig with the noise model
        emulator_config = EmulatorConfig(noise_model=noise_model)
        return emulator_config

    def get_default_submodule(self, option: str) -> None:
        """
        Raises ValueError as this module has no submodules.

        :param option: Option name
        :raises ValueError: If called, since this module has no submodules
        """
        raise ValueError("This module has no submodules.")
