import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from modules.applications.qml.generative_modeling.transformations.Transformation import Transformation
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
        input_data = {
            "best_sample": np.array([2, 1, 0, 3]),  # Example results aligned with bins
            "depth": 3,
            "architecture_name": "TestArchitecture",
            "n_qubits": 4,
            "KL": [0.1, 0.2, 0.05],
            "best_parameter": [0.5, 1.0],
            "circuit_transpiled": None,
            "store_dir_iter": "/tmp"
        }

        # Simulate the transformation configuration
        self.minmax_instance.transform_config = {
            "n_registers": 4
        }
        self.minmax_instance.histogram_train = np.array([0.1, 0.2])
        self.minmax_instance.histogram_train_original = np.array([0.05, 0.15])

        # Mock Transformation methods for alignment
        Transformation.compute_discretization_efficient = MagicMock(return_value=np.array([[0], [1], [2], [3]]))
        Transformation.generate_samples_efficient = MagicMock(return_value=np.array([[0], [1], [2], [3]]))

        # Call reverse_transform
        reversed_config = self.minmax_instance.reverse_transform(input_data)

        # Assertions
        self.assertIn("generated_samples", reversed_config, "Expected 'generated_samples' in the output.")
        self.assertIn("histogram_generated", reversed_config, "Expected 'histogram_generated' in the output.")

