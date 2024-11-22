import unittest
import numpy as np
from modules.applications.qml.generative_modeling.transformations.PIT import PIT


class TestPIT(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pit_instance = PIT()
        cls.sample_input_data = {
            "dataset_name": "TestDataset",
            "dataset": np.random.uniform(0, 1, (100, 4)),  # 100 samples with 4 features
            "n_qubits": 4,
            "store_dir_iter": "/tmp/pit_test",
            "train_size": 0.8,
        }
        cls.sample_config = {}

    # def test_transform(self):
    #     transformed_data = self.pit_instance.transform(self.sample_input_data, self.sample_config)
    #     self.assertIn("histogram_train", transformed_data)
    #     self.assertIn("binary_train", transformed_data)
    #     self.assertEqual(transformed_data["n_qubits"], self.sample_input_data["n_qubits"])
    #     self.assertAlmostEqual(
    #         np.sum(transformed_data["histogram_train"]), 1.0, places=5,
    #         msg="Histogram should sum to 1 after normalization."
    #     )

    # def test_reverse_transform(self):
    #     transformed_data = self.pit_instance.transform(self.sample_input_data, self.sample_config)
    #     reverse_input = {
    #         "best_sample": np.random.randint(0, 2 ** self.sample_input_data["n_qubits"], (50,)),
    #         "depth": 3,
    #         "architecture_name": "TestArchitecture",
    #         "n_qubits": self.sample_input_data["n_qubits"],
    #         "KL": [0.1, 0.2, 0.05],
    #         "circuit_transpiled": "MockCircuit",
    #         "store_dir_iter": self.sample_input_data["store_dir_iter"],
    #     }
    #     reversed_data = self.pit_instance.reverse_transform(reverse_input)
    #     self.assertIn("generated_samples", reversed_data)
    #     self.assertIn("histogram_generated_original", reversed_data)
    #     self.assertAlmostEqual(
    #         np.sum(reversed_data["histogram_generated_original"]), 1.0, places=5,
    #         msg="Reversed histogram should sum to 1."
    #     )

    def test_empirical_transformation(self):
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


if __name__ == "__main__":
    unittest.main()
