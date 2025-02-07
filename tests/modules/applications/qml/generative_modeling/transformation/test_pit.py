import unittest
import numpy as np
from unittest.mock import MagicMock

from src.modules.applications.qml.generative_modeling.transformations.pit import PIT
from src.modules.applications.qml.generative_modeling.circuits.circuit_copula import CircuitCopula


class TestPIT(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pit_instance = PIT()
        cls.sample_input_data = {
            "dataset_name": "test_dataset",
            "dataset": np.array([[1, 2], [3, 4], [5, 6]]),
            "n_qubits": 4,
            "store_dir_iter": "/tmp",
            "train_size": 0.8
        }
        cls.sample_config = {}
        # Mock reverse_epit_lookup for testing
        cls.pit_instance.reverse_epit_lookup = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2]
        ])
        cls.pit_instance.grid_shape = (2, 2)
        cls.pit_instance.transform_config = {
            "n_registers": 2,
            "binary_train": np.array([[0, 1], [1, 0]]),
            "histogram_train": np.array([0.5, 0.5]),
            "dataset_name": "mock_dataset",
            "store_dir_iter": "/mock/path"
        }

    def test_get_requirements(self):
        requirements = self.pit_instance.get_requirements()
        expected_requirements = [
            {"name": "numpy", "version": "1.26.4"},
            {"name": "pandas", "version": "2.2.3"}
        ]
        self.assertEqual(requirements, expected_requirements)

    def test_get_default_submodule(self):
        submodule = self.pit_instance.get_default_submodule("CircuitCopula")
        self.assertIsInstance(submodule, CircuitCopula, "Expected CircuitCopula instance for 'CircuitCopula' option.")
        with self.assertRaises(NotImplementedError):
            self.pit_instance.get_default_submodule("InvalidSubmodule")

    def test_transform(self):
        result = self.pit_instance.transform(self.sample_input_data, self.sample_config)

        self.assertIn("histogram_train", result, "Expected 'histogram_train' in transformation result.")
        self.assertIn("binary_train", result, "Expected 'binary_train' in transformation result.")
        self.assertIn("dataset_name", result, "Expected 'dataset_name' in transformation result.")
        self.assertEqual(result["dataset_name"], "test_dataset", "Dataset name mismatch in transformation result.")
        self.assertIsInstance(result["binary_train"], np.ndarray, "Expected binary_train to be a numpy array.")
        self.assertEqual(result["n_qubits"], self.sample_input_data["n_qubits"], "n_qubits mismatch.")

    # This test is currently commented out because:
    # - The `reverse_transform` method relies on mocked internal methods (`compute_discretization_efficient` and
    #   `generate_samples_efficient`) that require precise mocking of their behavior and returned data.
    # - Creating realistic mock data for `reverse_transform` is challenging without deeper understanding of
    #   the expected transformations or how they interact with the architecture.
    # - We plan to implement this test in the future when there is more clarity on the expected functionality
    # def test_reverse_transform(self):
    #     # Mocked input data
    #     input_data = {
    #         "best_sample": np.array([0, 1, 2, 3]),
    #         "depth": 2,
    #         "architecture_name": "TestArchitecture",
    #         "n_qubits": 2,
    #         "KL": [0.1, 0.2],
    #         "circuit_transpiled": None,
    #         "best_parameter": [0.5, 0.6],
    #         "store_dir_iter": "/mock/path"
    #     }

    #     # Mock internal method responses
    #     self.pit_instance.compute_discretization_efficient = MagicMock(return_value=np.array([[0, 1], [2, 3]]))
    #     self.pit_instance.generate_samples_efficient = MagicMock(return_value=np.array([[0.1, 0.2], [0.3, 0.4]]))

    #     # Call the method
    #     reverse_config = self.pit_instance.reverse_transform(input_data)

    #     # Validate the response
    #     self.assertIn("generated_samples", reverse_config)
    #     self.assertIn("transformed_samples", reverse_config)
    #     self.assertIn("KL_best_transformed", reverse_config)
    #     self.assertEqual(reverse_config["depth"], input_data["depth"])
    #     self.assertEqual(reverse_config["dataset_name"], self.pit_instance.dataset_name)

    def test_emp_integral_trans(self):
        data = np.random.uniform(0, 1, 100)
        transformed_data = self.pit_instance.emp_integral_trans(data)
        self.assertTrue((transformed_data >= 0).all() and (transformed_data <= 1).all(),
                        "Empirical transformation should map values to [0, 1].")

    def test_fit_transform(self):
        data = np.random.uniform(0, 1, (100, 4))
        transformed = self.pit_instance.fit_transform(data)
        self.assertEqual(transformed.shape, data.shape, "Transformed data should match the input shape.")

    def test_inverse_transform(self):
        data = np.random.uniform(0, 1, (100, 4))
        self.pit_instance.fit_transform(data)
        inverse_data = self.pit_instance.inverse_transform(data)
        self.assertEqual(inverse_data.shape, data.shape, "Inverse-transformed data should match the input shape.")

    # This test is currently commented out because:
    # We plan to revisit this test in the future
    # def test_reverse_empirical_integral_trans_single(self):
    #     self.pit_instance.reverse_epit_lookup = np.array([
    #         [0.1, 0.2, 0.3],
    #         [0.4, 0.5, 0.6]
    #     ])
    #     values = np.array([0.2, 0.8])
    #     reverse_result = self.pit_instance._reverse_emp_integral_trans_single(values)
    #     self.assertEqual(len(reverse_result), 1, "Reverse transformed result length mismatch.")
