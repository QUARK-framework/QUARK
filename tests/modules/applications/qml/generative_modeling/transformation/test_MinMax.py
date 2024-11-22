import unittest
import numpy as np
from modules.applications.qml.generative_modeling.transformations.MinMax import MinMax
from modules.applications.qml.generative_modeling.circuits.CircuitStandard import CircuitStandard
from modules.applications.qml.generative_modeling.circuits.CircuitCardinality import CircuitCardinality


class TestMinMax(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.minmax_instance = MinMax()
        cls.sample_input_data = {
            "dataset_name": "TestDataset",
            "dataset": np.random.rand(100, 3),  # 100 samples, 3 features
            "n_qubits": 6,
            "train_size": 0.8,
            "store_dir_iter": "/tmp/test"
        }
        cls.sample_config = {}

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

    # def test_transform(self):
    #     transformed_data = self.minmax_instance.transform(self.sample_input_data, self.sample_config)

    #     self.assertIn("histogram_train", transformed_data)
    #     self.assertIn("binary_train", transformed_data)
    #     self.assertIn("dataset_name", transformed_data)
    #     self.assertEqual(transformed_data["dataset_name"], self.sample_input_data["dataset_name"])
    #     self.assertEqual(transformed_data["n_qubits"], self.sample_input_data["n_qubits"])

    #     # Check histogram normalization
    #     histogram_train = transformed_data["histogram_train"]
    #     self.assertAlmostEqual(np.sum(histogram_train), 1.0, places=5, msg="Histogram values should sum to 1.")

    #     # Check binary_train shape
    #     binary_train = transformed_data["binary_train"]
    #     self.assertEqual(binary_train.shape[1], self.sample_input_data["n_qubits"])

    # def test_reverse_transform(self):
    #     # Ensure dataset contains numeric data
    #     dataset_size = 100
    #     n_qubits = self.sample_input_data["n_qubits"]
    #     numeric_dataset = np.random.uniform(0, 1, (dataset_size, n_qubits))

    #     # Update the input data with numeric dataset
    #     self.sample_input_data["dataset"] = numeric_dataset

    #     # Ensure the config matches the dataset
    #     self.sample_input_data["train_size"] = 0.8
    #     self.sample_config = {
    #         "store_dir_iter": self.sample_input_data["store_dir_iter"]
    #     }

    #     # Transform the data
    #     transformed_data = self.minmax_instance.transform(self.sample_input_data, self.sample_config)

    #     # Prepare reverse input for reverse_transform
    #     reverse_input = {
    #         "best_sample": np.random.randint(0, 2 ** n_qubits, (50,)),
    #         "depth": 3,
    #         "architecture_name": "TestArchitecture",
    #         "n_qubits": n_qubits,
    #         "KL": [0.1, 0.2, 0.05],
    #         "circuit_transpiled": "MockCircuit",
    #         "store_dir_iter": self.sample_input_data["store_dir_iter"]
    #     }

    #     # Test reverse_transform
    #     reversed_data = self.minmax_instance.reverse_transform(reverse_input)

    #     # Assertions to verify reversed data
    #     self.assertIn("generated_samples", reversed_data)
    #     self.assertIn("histogram_generated", reversed_data)
    #     self.assertIn("histogram_generated_original", reversed_data)
    #     self.assertAlmostEqual(
    #         np.sum(reversed_data["histogram_generated"]), 1.0,
    #         places=5, msg="Reversed histogram should sum to 1."
    #     )

    def test_fit_transform(self):
        sample_data = np.random.rand(50, 3)
        transformed_data = self.minmax_instance.fit_transform(sample_data)

        self.assertTrue(np.all(transformed_data >= 0))
        self.assertTrue(np.all(transformed_data <= 1))

    def test_inverse_transform(self):
        sample_data = np.random.rand(50, 3)
        transformed_data = self.minmax_instance.fit_transform(sample_data)
        inverse_transformed = self.minmax_instance.inverse_transform(transformed_data)


if __name__ == "__main__":
    unittest.main()
