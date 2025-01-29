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

from typing import Union
import numpy as np

from quark.modules.applications.qml.generative_modeling.transformations.Transformation import Transformation
from quark.modules.applications.qml.generative_modeling.circuits.CircuitStandard import CircuitStandard
from quark.modules.applications.qml.generative_modeling.circuits.CircuitCardinality import CircuitCardinality


class MinMax(Transformation):  # pylint: disable=R0902
    """
    In min-max normalization each data point is shifted
    such that it lies between 0 and 1.
    """

    def __init__(self):
        super().__init__("MinMax")
        self.submodule_options = ["CircuitStandard", "CircuitCardinality"]
        self.transform_config = None
        self.max = None
        self.min = None
        self.n_qubits = None
        self.dataset = None
        self.dataset_name = None
        self.grid_shape = None
        self.histogram_train = None
        self.histogram_train_original = None

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module.

        :return: List of dict with requirements of this module
        """
        return [{"name": "numpy", "version": "1.26.4"}]

    def get_default_submodule(self, option: str) -> Union[CircuitStandard, CircuitCardinality]:
        if option == "CircuitStandard":
            return CircuitStandard()
        elif option == "CircuitCardinality":
            return CircuitCardinality()
        else:
            raise NotImplementedError(f"Circuit Option {option} not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns empty dict as this transformation has no configurable settings.

        :return: Empty dict
        """
        return {}

    def transform(self, input_data: dict, config: dict) -> dict:
        """
        Transforms the input dataset using MinMax transformation and computes histograms
        of the training dataset in the transformed space.

        :param input_data: A dictionary containing information about the dataset and application configuration.
        :param config: A dictionary with parameters specified in the Config class.
        :return: A tuple containing a dictionary with MinMax-transformed data.
        """
        self.dataset_name = input_data["dataset_name"]
        self.dataset = input_data["dataset"]
        self.n_qubits = input_data["n_qubits"]
        self.grid_shape = int(2 ** (self.n_qubits // 2))
        n_registers = self.dataset.shape[-1]

        # Calculate ranges for the original dataset and the transformed dataset
        ranges_original = np.column_stack((np.min(self.dataset, axis=0), np.max(self.dataset, axis=0)))
        transformed_dataset = self.fit_transform(self.dataset)
        ranges_transformed = np.column_stack((np.min(transformed_dataset, axis=0), np.max(transformed_dataset, axis=0)))

        # Compute histogram for the transformed dataset
        transformed_histogram_grid = np.histogramdd(
            transformed_dataset,
            bins=self.grid_shape,
            range=ranges_transformed)[0]
        histogram_transformed_1d = transformed_histogram_grid.flatten()

        solution_space = np.zeros(len(transformed_dataset), dtype=int)
        # Initialize a variable to keep track of the current position in the result_array
        position = 0
        value = 0
        for count in histogram_transformed_1d:
            if count > 0:
                solution_space[position:position + int(count)] = value
                position += int(count)
            value += 1

        binary_strings = [np.binary_repr(x, width=self.n_qubits) for x in solution_space]
        binary_transformed = np.array([list(map(int, s)) for s in binary_strings])
        learned_histogram = np.histogramdd(self.dataset, bins=self.grid_shape, range=ranges_original)
        self.histogram_train_original = learned_histogram[0] / np.sum(learned_histogram[0])

        # Compute histogram for the transformed dataset
        learned_histogram = np.histogramdd(transformed_dataset, bins=self.grid_shape, range=ranges_transformed)
        histogram_train = learned_histogram[0] / np.sum(learned_histogram[0])
        self.histogram_train = histogram_train.flatten()

        self.transform_config = {
            "histogram_train": self.histogram_train,
            "binary_train": binary_transformed,
            "dataset_name": self.dataset_name,
            "n_registers": n_registers,
            "n_qubits": self.n_qubits,
            "train_size": input_data["train_size"],
            "store_dir_iter": input_data["store_dir_iter"]
        }

        return self.transform_config

    def reverse_transform(self, input_data: dict) -> dict:
        """
        Transforms the solution back to the representation needed for validation/evaluation.

        :param input_data: Dictionary containing the solution
        :return: Solution transformed accordingly
        """
        best_results = input_data["best_sample"]
        depth = input_data["depth"]
        architecture_name = input_data["architecture_name"]
        n_qubits = input_data["n_qubits"]
        n_registers = self.transform_config["n_registers"]
        KL_best_transformed = min(input_data["KL"])
        circuit_transpiled = input_data['circuit_transpiled']

        array_bins = Transformation.compute_discretization_efficient(n_qubits, n_registers)
        transformed_samples = Transformation.generate_samples_efficient(best_results, array_bins, n_registers,
                                                                        noisy=True)

        # Calculate ranges for the transformed samples
        ranges_transformed = np.column_stack((np.min(transformed_samples, axis=0), np.max(transformed_samples, axis=0)))

        # Compute histogram for the transformed samples
        learned_histogram = np.histogramdd(transformed_samples, bins=self.grid_shape, range=ranges_transformed)
        histogram_generated_transformed = learned_histogram[0] / np.sum(learned_histogram[0])
        histogram_generated_transformed = histogram_generated_transformed.flatten()

        original_samples = self.inverse_transform(transformed_samples)

        # Calculate ranges for the original samples
        ranges_original = np.column_stack((np.min(original_samples, axis=0), np.max(original_samples, axis=0)))

        # Compute histogram for the original samples
        learned_histogram = np.histogramdd(original_samples, bins=self.grid_shape, range=ranges_original)
        histogram_generated_original = learned_histogram[0] / np.sum(learned_histogram[0])
        histogram_generated_original = histogram_generated_original.flatten()

        best_parameter = input_data["best_parameter"]

        reverse_config_trans = {
            "generated_samples": best_results,
            "transformed_samples": transformed_samples,
            "depth": depth,
            "architecture_name": architecture_name,
            "dataset_name": self.dataset_name,
            "n_qubits": n_qubits,
            "best_parameter": best_parameter,
            "histogram_train": self.histogram_train,
            "histogram_train_original": self.histogram_train_original,
            "histogram_generated_original": histogram_generated_original,
            "histogram_generated": histogram_generated_transformed,
            "KL_best_transformed": KL_best_transformed,
            "store_dir_iter": input_data["store_dir_iter"],
            "circuit_transpiled": circuit_transpiled
        }

        return reverse_config_trans

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Method that performs the min max normalization.

        :param data: Data to be fitted
        :return: Fitted data
        """
        data_min = data.min()
        data_max = data.max() - data_min
        data = (data - data_min) / data_max

        return data

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Method that performs the inverse min max normalization.

        :param data: Data to be fitted
        :return: Data in original space
        """
        data_min = data.min()
        data_max = data.max() - data_min

        return data * data_max + data_min
