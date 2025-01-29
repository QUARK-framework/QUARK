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

import glob
import argparse
import unittest

from quark.__main__ import handle_benchmark_run, create_benchmark_parser


class TestMain(unittest.TestCase):

    def test_handle_valid_benchmark_run(self) -> None:
        """
        Test a couple of valid QUARK configs
        :return:
        :rtype: None
        """
        parser = argparse.ArgumentParser()
        create_benchmark_parser(parser)

        for filename in glob.glob("tests/configs/valid/**.yml"):
            with self.subTest(msg=f"Running Benchmark Test with valid config {filename}", filename=filename):
                args = parser.parse_args(["--config", filename, "--failfast"])
                self.assertEqual(handle_benchmark_run(args), None)

    def test_handle_invalid_benchmark_run(self) -> None:
        """
        Test a couple of invalid QUARK configs
        :return:
        :rtype: None
        """
        parser = argparse.ArgumentParser()
        create_benchmark_parser(parser)

        for filename in glob.glob("tests/configs/invalid/**.yml"):
            with self.subTest(msg=f"Running Benchmark Test with invalid config {filename}", filename=filename):
                args = parser.parse_args(["--config", filename, "--failfast"])
                with self.assertRaises(Exception):
                    handle_benchmark_run(args)
