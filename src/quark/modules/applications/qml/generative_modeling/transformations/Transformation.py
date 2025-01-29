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

from itertools import product
from abc import ABC, abstractmethod

import numpy as np

from quark.modules.Core import Core
from quark.utils import start_time_measurement, end_time_measurement


class Transformation(Core, ABC):
    """
    The task of the transformation module is to translate data and problem
    specification of the application into preprocessed format.
    """

    def __init__(self, name):
        """
        Constructor method.
        """
        super().__init__()
        self.transformation_name = name

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module.

        :return: List of dict with requirements of this module
        """
        return [{"name": "numpy", "version": "1.26.4"}]

    def preprocess(self, input_data: dict, config: dict, **kwargs: dict) -> tuple[dict, float]:
        """
        In this module, the preprocessing step is transforming the data to the correct target format.

        :param input_data: Collected information of the benchmarking process
        :param config: Config specifying the parameters of the transformation
        :param kwargs: Additional optional arguments
        :return: Tuple with transformed problem and the time it took to map it
        """
        start = start_time_measurement()
        output = self.transform(input_data, config)

        return output, end_time_measurement(start)

    def postprocess(self, input_data: dict, config: dict, **kwargs) -> tuple[dict, float]:
        """
        Does the reverse transformation.

        :param input_data: Dictionary containing information of previously executed modules
        :param config: Dictionary containing additional information
        :param kwargs: Dictionary containing additional information
        :return: Tuple with the dictionary and the time the postprocessing took
        """
        start = start_time_measurement()
        output = self.reverse_transform(input_data)
        output["Transformation"] = True
        if "inference" in input_data:
            output["inference"] = input_data["inference"]

        return output, end_time_measurement(start)

    @abstractmethod
    def transform(self, input_data: dict, config: dict) -> dict:
        """
        Helps to ensure that the model can effectively learn the underlying
        patterns and structure of the data, and produce high-quality outputs.

        :param input_data: Input data for transformation
        :param config: Configuration parameters for the transformation
        :return: Transformed data.
        """
        return input_data

    def reverse_transform(self, input_data: dict) -> dict:
        """
        Transforms the solution back to the original problem.
        This might not be necessary in all cases, so the default is to return the original solution.
        This might be needed to convert the solution to a representation needed for validation and evaluation.

        :param input_data: The input data to be transformed
        :return: Transformed data
        """
        return input_data

    @staticmethod
    def compute_discretization(n_qubits: int, n_registered: int) -> np.ndarray:
        """
        Compute discretization for the grid.

        :param n_qubits: Total number of qubits
        :param n_registered: Number of qubits to be registered
        :return: Discretization data
        """
        n = 2 ** (n_qubits // n_registered)
        n_bins = n ** n_registered
        bin_data = np.empty((n_bins, n_registered + 1), dtype=float)

        for k, coords in enumerate(product(range(n), repeat=n_registered)):
            normalized_coords = np.array(coords) / n + 0.5 / n
            bin_data[k] = np.concatenate(([k], normalized_coords))

        return bin_data

    @staticmethod
    def compute_discretization_efficient(n_qubits: int, n_registers: int) -> np.ndarray:
        """
        Compute grid discretization.

        :param n_qubits: Total number of qubits
        :param n_registers: Number of qubits to be registered
        :return: Discretization data
        """
        n = 2 ** (n_qubits // n_registers)
        n_bins = n ** n_registers

        # Create an array of all possible coordinate combinations
        coords = np.array(list(product(range(n), repeat=n_registers)))

        # Calculate normalized_coords for all combinations
        normalized_coords = (coords.astype(float) + 0.5) / n

        # Create bin_data by concatenating normalized_coords with row indices
        bin_data = np.hstack((np.arange(n_bins).reshape(-1, 1), normalized_coords))

        return bin_data

    @staticmethod
    def generate_samples(results: np.ndarray, bin_data: np.ndarray, n_registers: int, noisy: bool = True) -> np.ndarray:
        """
        Generate samples based on measurement results and the grid bins.

        :param results: Results of measurements
        :param bin_data: Binned data
        :param n_registers: Number of registers
        :param noisy: Flag indicating whether to add noise
        :return: Generated samples
        """
        n_shots = np.sum(results)
        width = 1 / len(bin_data) ** (1 / n_registers)
        points = (
            0.5 * width * np.random.uniform(low=-1, high=1, size=(n_shots, n_registers))
            if noisy
            else np.zeros((n_shots, n_registers))
        )

        position = 0
        for idx, value in enumerate(results):
            bin_coords = bin_data[idx, 1:]
            points[position: position + value, :] += np.tile(bin_coords, (value, 1))
            position += value

        return points.astype(np.float32)

    @staticmethod
    def generate_samples_efficient(results, bin_data: np.ndarray, n_registers: int, noisy: bool = True) -> np.ndarray:
        """
        Generate samples efficiently using numpy arrays based on measurement results and the grid bins.

        :param results: Results of measurements
        :param bin_data: Binned data
        :param n_registers: Number of registers
        :param noisy: Flag indicating whether to add noise
        :return: Generated samples
        """
        n_shots = np.sum(results)
        width = 1 / len(bin_data) ** (1 / n_registers)

        # Generate random noise or zeros
        noise = (
            0.5 * width * np.random.uniform(low=-1, high=1, size=(n_shots, n_registers))
            if noisy
            else np.zeros((n_shots, n_registers))
        )

        # Create an array of bin_coords for each result, then stack them vertically
        bin_coords = bin_data[:, 1:]
        expanded_bin_coords = np.repeat(bin_coords, results, axis=0)

        # Reshape expanded_bin_coords to match the shape of noise
        expanded_bin_coords = expanded_bin_coords.reshape(n_shots, n_registers)

        # Add noise to the expanded_bin_coords
        points = expanded_bin_coords + noise

        return points.astype(np.float32)
