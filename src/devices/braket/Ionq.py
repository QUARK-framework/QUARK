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

from braket.aws import AwsDevice

from devices.braket.Braket import Braket


class Ionq(Braket):
    """
    Class for using the IonQ devices on Amazon Braket
    """

    def __init__(self, device_name: str, device_arn: str = 'arn:aws:braket:::device/qpu/ionq/ionQdevice'):
        """
        Constructor method
        """
        super().__init__(region="us-east-1", device_name=device_name, arn=device_arn)
        self.init_s3_storage("ionq")
        self.device = AwsDevice(device_arn, aws_session=self.aws_session)
