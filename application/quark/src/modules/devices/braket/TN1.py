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

import os
from braket.aws import AwsDevice

from modules.devices.braket.Braket import Braket
from modules.Core import Core


class TN1(Braket):
    """
    Class for using the TN1 simulator on Amazon Braket.
    """

    def __init__(self, device_name: str, arn: str = 'arn:aws:braket:::device/quantum-simulator/amazon/tn1'):
        """
        Constructor method.

        :param device_name: Name of the device
        :param arn: Amazon resource name for the TN1 simulator
        """
        super().__init__(device_name=device_name, arn=arn)
        self.submodule_options = []

        if 'SKIP_INIT' in os.environ:
            # TODO: This is currently needed so create_module_db in the Installer does not execute the rest
            # of this section, which would be unnecessary. However, this should be done better in the future!
            return

        self.init_s3_storage("tn1")
        self.device = AwsDevice(arn, aws_session=self.aws_session)

    def get_parameter_options(self) -> dict:
        """
        Returns empty dict as this solver has no configurable settings.

        :return: Empty dict
        """
        return {}

    def get_default_submodule(self, option: str) -> None:
        """
        Raises ValueError as this module has no submodules.

        :param option: Option name
        :raises ValueError: If called, since this module has no submodules
        """
        raise ValueError("This module has no submodules.")
