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

import numpy as np
import pandas as pd

from quark.modules.applications.qml.generative_modeling.transformations.Transformation import Transformation
from quark.modules.applications.qml.generative_modeling.circuits.CircuitCopula import CircuitCopula


class PIT(Transformation):  # pylint disable=R0902
    """
    The transformation of the original probability distribution to
    the distribution of its uniformly distributed cumulative marginals is known as the copula.
    """

    def __init__(self):
        super().__init__("PIT")
        self.submodule_options = ["CircuitCopula"]
        self.reverse_epit_lookup = None
        self.transform_config = None
        self.n_qubits = None
        self.dataset = None
        self.dataset_name = None
        self.grid_shape = None
        self.histogram_train = None
        self.histogram_train_original = None
        self.histogram_transformed = None

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module.

        :return: List of dict with requirements of this module
        """
        return [
            {"name": "numpy", "version": "1.26.4"},
            {"name": "pandas", "version": "2.2.3"}
        ]

    def get_parameter_options(self) -> dict:
        """
        Returns empty dict as this transformation has no configurable settings.

        :return: Empty dict
        """
        return {}

    def get_default_submodule(self, option: str) -> CircuitCopula:
        if option == "CircuitCopula":
            return CircuitCopula()
        else:
            raise NotImplementedError(f"Circuit Option {option} not implemented")

    def transform(self, input_data: dict, config: dict) -> dict:
        """
        Transforms the input dataset using PIT transformation and computes histograms
        of the training dataset in the transformed space.

        :param input_data: Dataset
        :param config: Config with the parameters specified in Config class
        :return: Dict with PIT transformation, time it took to map it
        """
        # TODO: PIT.transform is almost identical to MinMax.transform -> function should be moved to Transformation.py
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
        self.histogram_transformed = histogram_transformed_1d / np.sum(histogram_transformed_1d)

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
        # Convert the binary strings to a NumPy array of integers
        binary_transformed = np.array([list(map(int, s)) for s in binary_strings])

        # Compute histogram for the original dataset
        learned_histogram = np.histogramdd(self.dataset, bins=self.grid_shape, range=ranges_original)
        self.histogram_train_original = learned_histogram[0] / np.sum(learned_histogram[0])

        # Compute histogram for the transformed dataset
        train_histogram = np.histogramdd(transformed_dataset, bins=self.grid_shape, range=ranges_transformed)
        histogram_train = train_histogram[0] / np.sum(train_histogram[0])
        self.histogram_train = histogram_train.flatten()

        self.transform_config = {
            "histogram_train": self.histogram_train,
            "binary_train": binary_transformed,
            "dataset_name": self.dataset_name,
            "n_registers": n_registers,
            "n_qubits": self.n_qubits,
            "store_dir_iter": input_data["store_dir_iter"],
            "train_size": input_data["train_size"],
            "transformed_dataset": transformed_dataset
        }

        return self.transform_config

    def reverse_transform(self, input_data: dict) -> dict:
        """
        Transforms the solution back to the representation needed for validation/evaluation.

        :param input_data: Dictionary containing the solution
        :return: Dictionary with solution transformed accordingly
        """
        depth = input_data["depth"]
        architecture_name = input_data["architecture_name"]
        n_qubits = input_data["n_qubits"]
        n_registers = self.transform_config["n_registers"]
        kl_best_transformed = min(input_data["KL"])
        best_results = input_data["best_sample"]
        circuit_transpiled = input_data['circuit_transpiled']

        array_bins = self.compute_discretization_efficient(n_qubits, n_registers)
        transformed_samples = self.generate_samples_efficient(best_results, array_bins, n_registers, noisy=True)

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
            "histogram_train_original": self.histogram_train_original,
            "histogram_train": self.histogram_train,
            "histogram_generated_original": histogram_generated_original,
            "histogram_generated": histogram_generated_transformed,
            "KL_best_transformed": kl_best_transformed,
            "store_dir_iter": input_data["store_dir_iter"],
            "circuit_transpiled": circuit_transpiled
        }

        return reverse_config_trans

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Takes the data points and applies the PIT.

        :param data: Data samples
        :return: Transformed data points
        """
        df = pd.DataFrame(data)
        epit = df.copy(deep=True).transpose()
        self.reverse_epit_lookup = epit.copy(deep=True)

        epit.values[::] = [self.emp_integral_trans(row) for row in epit.values]
        epit = epit.transpose()
        self.reverse_epit_lookup.values[::] = [np.sort(row) for row in self.reverse_epit_lookup.values]

        df = epit.copy()
        self.reverse_epit_lookup = self.reverse_epit_lookup.values
        return df.values

    def _reverse_emp_integral_trans_single(self, values: np.ndarray) -> list[float]:
        """
        Takes one data point and applies the inverse PIT.

        :param values: Data point
        :return: Data point after applying the inverse transformation
        """
        values = values * (np.shape(self.reverse_epit_lookup)[1] - 1)
        rows = np.shape(self.reverse_epit_lookup)[0]

        # TODO THIS IS A TEMPORARY BUG FIX FOR RARELY OCCURRING CASES THAT LEAD TO INDEX_ERRORS.
        #  WILL BE FIXED IN RELEASE 2.1.3.
        # Ensure values are within bounds
        values_l = np.clip(np.floor(values).astype(int), 0, self.reverse_epit_lookup.shape[1] - 1)
        values_h = np.clip(np.ceil(values).astype(int), 0, self.reverse_epit_lookup.shape[1] - 1)

        # if we are an integer then floor and ceiling are the same
        is_int_mask = 1 - (values_h - values_l)
        row_indexer = np.arange(rows)
        result_l = self.reverse_epit_lookup[
            ([row_indexer], [values_l])]  # doing 2d lookup as [[index1.row, index2.row],[index1.column, index2.column]]
        result_h = self.reverse_epit_lookup[
            ([row_indexer], [values_h])]  # where 2d index tuple would be (index1.row, index1.column)
        # lookup int or do linear interpolation
        return result_l * (is_int_mask + values - values_l) + result_h * (values_h - values)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Applies the inverse transformation to the full data set.

        :param data: Data set
        :return: Data set after applying the inverse transformation
        """
        res = [self._reverse_emp_integral_trans_single(row) for row in data]

        return np.array(res)[:, 0, :]

    def emp_integral_trans(self, data: np.ndarray) -> np.ndarray:
        """
        Applies the empirical integral transformation to the given data.

        :param data: Data points
        :return: Empirically transformed data points
        """
        rank = np.argsort(data).argsort()
        length = data.size
        ecdf = np.linspace(0, 1, length, dtype=np.float64)
        ecdf_biject = ecdf[rank]
        return ecdf_biject
