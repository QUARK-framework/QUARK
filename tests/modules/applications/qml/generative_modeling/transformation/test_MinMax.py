import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from modules.applications.qml.generative_modeling.transformations.MinMax import MinMax
from modules.applications.qml.generative_modeling.circuits.CircuitStandard import CircuitStandard
from modules.applications.qml.generative_modeling.circuits.CircuitCardinality import CircuitCardinality


class TestMinMax(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.minmax_instance = MinMax()
        cls.sample_input_data = {
            "dataset_name": "TestDataset",
            "dataset": np.random.rand(100, 3),
            "n_qubits": 6,
            "train_size": 80,
            "store_dir_iter": "/tmp/test"
        }
        cls.sample_config = {}
        cls.minmax_instance.grid_shape = 4  # Example grid shape
        cls.minmax_instance.dataset_name = "TestDataset"
        cls.minmax_instance.histogram_train = np.array([0.2, 0.3, 0.1, 0.4])
        cls.minmax_instance.histogram_train_original = np.array([0.25, 0.25, 0.25, 0.25])
        cls.minmax_instance.transform_config = {"n_registers": 2}

    def test_get_requirements(self):
        requirements = self.minmax_instance.get_requirements()
        expected_requirements = [{"name": "numpy", "version": "1.26.4"}]
        self.assertEqual(requirements, expected_requirements)

    def test_get_default_submodule(self):
        submodule = self.minmax_instance.get_default_submodule("CircuitStandard")
        self.assertIsInstance(submodule, CircuitStandard)

        submodule = self.minmax_instance.get_default_submodule("CircuitCardinality")
        self.assertIsInstance(submodule, CircuitCardinality)

        with self.assertRaises(NotImplementedError):
            self.minmax_instance.get_default_submodule("InvalidOption")

    def test_fit_transform(self):
        data = np.array([[1, 2], [3, 4]])
        normalized = self.minmax_instance.fit_transform(data)
        self.assertTrue((normalized >= 0).all() and (normalized <= 1).all(), "Normalized data should be in [0, 1].")

    def test_inverse_transform(self):
        data = np.array([[0.5, 0.5], [0, 0]])
        original = self.minmax_instance.inverse_transform(data)
        self.assertTrue((original >= data.min()).all() and (original <= data.max()).all(),
                        "Reconstructed data should match the original range.")

    # @patch("modules.applications.qml.generative_modeling.transformations.MinMax")
    # def test_transform(self, mock_fit_transform):
    #     # Mock fit_transform to normalize the dataset to [0, 1]
    #     mock_fit_transform.side_effect = lambda x: (x - np.min(x, axis=0)) / np.ptp(x, axis=0)

    #     # Ensure solution_space values align with self.n_qubits
    #     self.minmax_instance.n_qubits = self.sample_input_data["n_qubits"]
    #     max_value = 2 ** self.minmax_instance.n_qubits - 1
        

    #     # Mock solution_space to be 1D and within the range of self.n_qubits
    #     solution_space = np.random.randint(0, max_value + 1, size=100)  # Flattened 1D array

    #     # Debug solution_space
    #     print("Generated solution_space:", solution_space)
    #     assert all(x < 2 ** self.minmax_instance.n_qubits for x in solution_space), \
    #         f"solution_space contains values exceeding {2 ** self.minmax_instance.n_qubits - 1}"

    #     # Intercept solution_space generation
    #     with patch("modules.applications.qml.generative_modeling.transformations.MinMax.transform") as mocked_transform:
    #         def side_effect(input_data, config):
    #             result = self.minmax_instance.transform(input_data, config)
    #             result["binary_train"] = np.array(
    #                 [list(map(int, np.binary_repr(x, width=self.minmax_instance.n_qubits))) for x in solution_space]
    #             )
    #             return result

    #         mocked_transform.side_effect = side_effect

    #         # Call the transform method
    #         result = self.minmax_instance.transform(self.sample_input_data, self.sample_config)

        
    #     # Validate result keys
    #     expected_keys = [
    #         "histogram_train", "binary_train", "dataset_name", "n_registers",
    #         "n_qubits", "train_size", "store_dir_iter"
    #     ]
    #     for key in expected_keys:
    #         self.assertIn(key, result, f"Key '{key}' is missing in the transform result.")

    #     # Validate transformed dataset
    #     transformed_dataset = mock_fit_transform(self.sample_input_data["dataset"])
    #     self.assertTrue(
    #         (transformed_dataset >= 0).all() and (transformed_dataset <= 1).all(),
    #         "Transformed dataset should be normalized to the range [0, 1]."
    #     )

    #     # Validate binary_train
    #     binary_train = result["binary_train"]
    #     self.assertIsInstance(binary_train, np.ndarray, "The 'binary_train' should be a NumPy array.")
    #     self.assertEqual(
    #         binary_train.shape[1],
    #         self.sample_input_data["n_qubits"],
    #         "Each binary vector in 'binary_train' should match the number of qubits."
    #     )
    #     self.assertTrue(
    #         np.isin(binary_train, [0, 1]).all(),
    #         "All values in 'binary_train' should be binary (0 or 1)."
    #     )

    #     # Validate histogram_train
    #     histogram_train = result["histogram_train"]
    #     self.assertAlmostEqual(
    #         np.sum(histogram_train),
    #         1,
    #         places=5,
    #         msg="The 'histogram_train' should be normalized to sum to 1."
    #     )
    #     self.assertTrue(
    #         (histogram_train >= 0).all(),
    #         "The 'histogram_train' should contain non-negative values."
    #     )

    #     # Validate dataset metadata
    #     self.assertEqual(result["dataset_name"], self.sample_input_data["dataset_name"], "Dataset name mismatch.")
    #     self.assertEqual(result["n_qubits"], self.sample_input_data["n_qubits"], "Number of qubits mismatch.")
    #     self.assertEqual(result["train_size"], self.sample_input_data["train_size"], "Train size mismatch.")
    #     self.assertEqual(result["store_dir_iter"], self.sample_input_data["store_dir_iter"], "Store directory mismatch.")
    # @patch("modules.applications.qml.generative_modeling.transformations.MinMax")
    # @patch("modules.applications.qml.generative_modeling.transformations.Transformation")
    # def test_reverse_transform(self, mock_generate_samples, mock_compute_discretization):
    #     """Test the reverse_transform method for correct behavior."""
    #     # Mock methods
    #     mock_compute_discretization.return_value = np.array([1, 2, 4])
    #     mock_generate_samples.return_value = np.random.rand(64, 2) 

    #     # Input data for reverse_transform
    #     input_data = {
    #         "best_sample": np.array([32, 32]),
    #         "depth": 3,
    #         "architecture_name": "TestArchitecture",
    #         "n_qubits": 6,
    #         "KL": [0.1, 0.2],
    #         "circuit_transpiled": MagicMock(),
    #         "best_parameter": np.array([0.3, 0.7]),
    #         "store_dir_iter": "./test_dir"
    #     }

    #     # Call reverse_transform
    #     result = self.minmax_instance.reverse_transform(input_data)

    #     # Validate result keys
    #     expected_keys = [
    #         "generated_samples", "transformed_samples", "depth", "architecture_name",
    #         "dataset_name", "n_qubits", "best_parameter", "histogram_train",
    #         "histogram_train_original", "histogram_generated_original",
    #         "histogram_generated", "KL_best_transformed", "store_dir_iter",
    #         "circuit_transpiled"
    #     ]
    #     for key in expected_keys:
    #         self.assertIn(key, result, f"Key '{key}' is missing in the reverse_transform result.")

    #     # Validate individual components
    #     self.assertEqual(result["depth"], input_data["depth"], "Depth mismatch in the result.")
    #     self.assertEqual(result["architecture_name"], input_data["architecture_name"], "Architecture name mismatch.")
    #     self.assertEqual(result["n_qubits"], input_data["n_qubits"], "Number of qubits mismatch.")
    #     self.assertEqual(result["dataset_name"], self.minmax_instance.dataset_name, "Dataset name mismatch.")
    #     np.testing.assert_array_equal(
    #         result["best_parameter"],
    #         input_data["best_parameter"],
    #         "Best parameter mismatch."
    #     )
    #     np.testing.assert_array_almost_equal(
    #         result["histogram_train"],
    #         self.minmax_instance.histogram_train,
    #         err_msg="Histogram train mismatch."
    #     )
    #     np.testing.assert_array_almost_equal(
    #         result["histogram_train_original"],
    #         self.minmax_instance.histogram_train_original,
    #         err_msg="Histogram train original mismatch."
    #     )
    #     self.assertEqual(
    #         result["KL_best_transformed"],
    #         min(input_data["KL"]),
    #         "KL best transformed mismatch."
    #     )