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

import getpass
import logging
import os
from abc import ABC
from datetime import datetime

import boto3
from botocore.config import Config
from botocore.exceptions import ProfileNotFound
from braket.aws import AwsSession

from quark.modules.devices.Device import Device


class Braket(Device, ABC):
    """
    Abstract class to use the Amazon Braket devices.
    """

    def __init__(self, device_name: str, region: str = None, arn: str = None):
        """
        Constructor method.
        """
        super().__init__(device_name)
        self.device = None
        self.arn = arn
        self.s3_destination_folder = None
        self.boto_session = None
        self.aws_session = None

        if 'SKIP_INIT' in os.environ:
            # TODO: This is currently needed so create_module_db in the Installer does not execute the rest
            #       of this section, which would be unnecessary. However, this should be done better in the future!
            return

        if device_name != "LocalSimulator":
            self._configure_aws_session(region)

    def _configure_aws_session(self, region: str) -> None:
        """
        Configures the AWS session for the Braket device.

        :param region: AWS region to use
        """
        proxy_definitions = self._setup_proxy()
        region = self._set_region(region)
        my_config = Config(region_name=region, proxies=proxy_definitions)

        profile_name = self._set_profile()
        self._initialize_aws_session(profile_name, region, my_config)

    @staticmethod
    def _setup_proxy() -> any:
        """
        Sets up proxy configuration if available in the environment variables.

        :return: Proxy definitions
        """
        if 'HTTP_PROXY' in os.environ:
            proxy_definitions = {
                'http': os.environ['HTTP_PROXY'],
                'https': os.environ['HTTP_PROXY']
            }
            os.environ['HTTPS_PROXY'] = os.environ['HTTP_PROXY']
        else:
            logging.warning('No HTTP_PROXY set as an environment variable. '
                            'This might cause trouble if you are using a VPN.')
            proxy_definitions = None
        return proxy_definitions

    @staticmethod
    def _set_region(region: str) -> str:
        """
        Sets the AWS region from the environment variable or defaults to 'us-east-1'.

        :param region: Provided region
        :return: Final region to be used
        """
        if region is None:
            region = os.environ.get('AWS_REGION', 'us-east-1')
            logging.info(f"No AWS_REGION specified, using default region: {region}")
        return region

    @staticmethod
    def _set_profile() -> str:
        """
        Determines the AWS profile to use for the session.

        :return: AWS profile name
        """
        if 'AWS_PROFILE' in os.environ:
            return os.environ['AWS_PROFILE']
        elif "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI" in os.environ:
            logging.info("Assuming AWS container environment, using container credentials.")
            return None
        else:
            profile_name = 'quantum_computing'
            os.environ['AWS_PROFILE'] = profile_name
            logging.info(f"No AWS_PROFILE specified, using default profile: {profile_name}")
            return profile_name

    def _initialize_aws_session(self, profile_name: str, region: str, my_config: Config) -> None:
        """
        Initializes the AWS session for interacting with Amazon Braket.

        :param profile_name: AWS profile name
        :param region: AWS region
        :param my_config: Boto3 configuration
        :raises Exception: If the AWS profile is not found
        """
        try:
            if profile_name is None:
                self.boto_session = boto3.Session(region_name=region)
            else:
                self.boto_session = boto3.Session(profile_name=profile_name, region_name=region)
            self.aws_session = AwsSession(boto_session=self.boto_session, config=my_config)
        except ProfileNotFound as exc:
            logging.error(f"AWS-Profile {profile_name} could not be found! Please set the AWS_PROFILE env variable. "
                          "Only LocalSimulator is available.")
            raise Exception("Please refer to the logged error message.") from exc

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module.

        :return: List of dictionaries with requirements
        """
        return [
            {"name": "amazon-braket-sdk", "version": "1.88.2"},
            {"name": "botocore", "version": "1.35.73"},
            {"name": "boto3", "version": "1.35.73"}
        ]

    def init_s3_storage(self, folder_name: str) -> None:
        """
        Initializes an S3 storage bucket for Amazon Braket.

        :param folder_name: Name of the s3 bucket
        """
        run_timestamp = datetime.today().date()
        username = getpass.getuser()

        bucket_name = f"amazon-braket-benchmark-framework-{run_timestamp}-{username}"
        self.s3_destination_folder = (bucket_name, folder_name)
        self._create_s3_bucket(self.boto_session, bucket_name)

    @staticmethod
    def _create_s3_bucket(boto3_session: boto3.Session, bucket_name: str = 'quark-benchmark-framework',
                          region: str = 'us-east-1') -> None:
        """
        Creates an S3 bucket with specific configurations.

        :param boto3-session: Boto3 session
        :param bucket_name: Name of the s3 bucket
        :param region: AWS region
        """
        s3_client = boto3_session.client('s3', region_name=region)

        if region == "us-east-1":
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            location = {"LocationConstraint": region}
            s3_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration=location)

        s3_client.put_public_access_block(
            Bucket=bucket_name,
            PublicAccessBlockConfiguration={
                'BlockPublicAcls': True,
                'IgnorePublicAcls': True,
                'BlockPublicPolicy': True,
                'RestrictPublicBuckets': True
            },
        )
