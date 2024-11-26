import unittest
import numpy as np
from unittest.mock import MagicMock

from modules.applications.qml.generative_modeling.transformations.PIT import PIT
from modules.applications.qml.generative_modeling.circuits.CircuitCopula import CircuitCopula


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
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
        ])
    
    def test_get_requirements(self):
        requirements = self.pit_instance.get_requirements()
        expected_requirements = [
            {"name": "numpy", "version": "1.26.4"},
            {"name": "pandas", "version": "2.2.2"}
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

    # def test_reverse_transform(self):
    #     # Mock transformation data
    #     self.pit_instance.transform_config = {
    #         "n_registers": 2,
    #     }
    #     input_data = {
    #         "depth": 3,
    #         "architecture_name": "test_architecture",
    #         "n_qubits": 4,
    #         "KL": [0.1, 0.05, 0.2],
    #         "best_sample": np.array([0, 1, 1, 0]),
    #         "circuit_transpiled": MagicMock(),
    #         "store_dir_iter": "/tmp",
    #         "best_parameter": np.random.random(4)
    #     }

    #     result = self.pit_instance.reverse_transform(input_data)

    #     self.assertIn("generated_samples", result, "Expected 'generated_samples' in reverse transform result.")
    #     self.assertIn("histogram_generated_original", result, "Expected 'histogram_generated_original' in result.")
    #     self.assertIn("KL_best_transformed", result, "Expected 'KL_best_transformed' in reverse transform result.")

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
    
    # def test_reverse_emp_integral_trans_single(self):
    #     # Case 1: Integer-like values
    #     values = np.array([0.2, 0.4])
    #     expected_result = [0.2, 0.5]
    #     result = self.pit_instance._reverse_emp_integral_trans_single(values)
    #     np.testing.assert_almost_equal(result, expected_result, decimal=6, err_msg="Failed for integer-like values")

    #     # Case 2: Non-integer values
    #     values = np.array([0.35, 0.55])
    #     expected_result = [0.275, 0.625]  # Interpolated values
    #     result = self.pit_instance._reverse_emp_integral_trans_single(values)
    #     np.testing.assert_almost_equal(result, expected_result, decimal=6, err_msg="Failed for non-integer values")

    #     # Case 3: Out-of-bounds values
    #     values = np.array([1.1, -0.1])
    #     try:
    #         self.pit_instance._reverse_emp_integral_trans_single(values)
    #         self.fail("Expected IndexError for out-of-bounds values")
    #     except IndexError:
    #         pass  # Test passes if IndexError is raised