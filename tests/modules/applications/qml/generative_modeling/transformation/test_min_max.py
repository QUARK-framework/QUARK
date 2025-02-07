import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from src.modules.applications.qml.generative_modeling.transformations.transformation import Transformation
from src.modules.applications.qml.generative_modeling.transformations.min_max import MinMax
from src.modules.applications.qml.generative_modeling.circuits.circuit_standard import CircuitStandard
from src.modules.applications.qml.generative_modeling.circuits.circuit_cardinality import CircuitCardinality


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

    def test_transform(self):
        input_data = {
            "dataset_name": "test_dataset",
            "dataset": np.array([[1, 2], [3, 4], [5, 6]]),
            "n_qubits": 4,
            "train_size": 0.8,
            "store_dir_iter": "/tmp"
        }
        config = {}

        transformed_config = self.minmax_instance.transform(input_data, config)

        self.assertIn("histogram_train", transformed_config, "Expected histogram_train in the transformed config.")
        self.assertIn("binary_train", transformed_config, "Expected binary_train in the transformed config.")
        self.assertEqual(transformed_config["dataset_name"], "test_dataset", "Dataset name should match.")
        self.assertEqual(transformed_config["n_qubits"], 4, "Expected number of qubits to match.")
        self.assertEqual(transformed_config["train_size"], 0.8, "Expected train size to match.")

    def test_reverse_transform(self):
        # self.minmax_instance.transform(self.sample_input_data, self.sample_config)

        # Mock the input for reverse_transform
        reverse_input_data = {
            "best_sample": np.array([0, 1, 2]),  # Example best samples
            "depth": 1,  # Example
            "architecture_name": "test_arch",  # Example
            "n_qubits": 6,
            "KL": [0.1],  # Example
            "best_parameter": [0.5],
            "store_dir_iter": "test_dir",
            "circuit_transpiled": MagicMock() # Mock the circuit
        }

        reverse_config = self.minmax_instance.reverse_transform(reverse_input_data)

        self.assertIn("generated_samples", reverse_config)
