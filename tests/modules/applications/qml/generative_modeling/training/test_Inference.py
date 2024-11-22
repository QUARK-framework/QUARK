import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from modules.applications.qml.generative_modeling.training.Inference import Inference


class TestInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.inference_instance = Inference()
        cls.mock_parameters = np.array([0.5, 0.2, 0.3])  # Example parameters
        cls.config = {"pretrained": "mock_pretrained_file.npy"}
        cls.input_data = {
            "n_qubits": 3,
            "histogram_train": np.array([0.1, 0.2, 0.3, 0.4]),
            "execute_circuit": MagicMock(),
            "n_shots": 1000,
        }

    @patch("numpy.load")
    def test_start_training(self, mock_np_load):
        # Mock np.load to return mock parameters
        mock_np_load.return_value = self.mock_parameters

        # Update the execute_circuit mock to return a PMF with the correct size
        n_states = 2 ** self.input_data["n_qubits"]
        pmf_mock = np.full(n_states, 1 / n_states)  # Uniform distribution over all states
        self.input_data["execute_circuit"].return_value = ([pmf_mock], None)

        # Call the start_training method
        result = self.inference_instance.start_training(self.input_data, self.config)

       # Extract KL divergence
        kl_values = result["KL"]

        # Aggregate KL values (e.g., take the mean or first element)
        kl_value = np.mean(kl_values)  # Use mean for validation

        # Validate KL divergence
        self.assertGreaterEqual(kl_value, 0, "KL divergence should be non-negative.")

        # Validate the results
        self.assertIn("best_parameter", result)
        self.assertIn("inference", result)
        self.assertIn("KL", result)
        self.assertIn("best_sample", result)

        # Validate best sample
        self.assertTrue((result["best_sample"] >= 0).all(), "Best sample should contain non-negative integers.")

    def test_get_requirements(self):
        requirements = self.inference_instance.get_requirements()
        expected_requirements = [{"name": "numpy", "version": "1.26.4"}]
        self.assertEqual(requirements, expected_requirements)

    def test_get_parameter_options(self):
        parameter_options = self.inference_instance.get_parameter_options()
        expected_options = {
            "pretrained": {
                "values": [],
                "custom_input": True,
                "postproc": str,
                "description": "Please provide the parameters of a pretrained model.",
            }
        }
        self.assertEqual(parameter_options, expected_options)

    def test_get_default_submodule(self):
        with self.assertRaises(ValueError):
            self.inference_instance.get_default_submodule("any_option")

    # @patch("numpy.asarray")
    # def test_kl_divergence(self, mock_np_asarray):
    #     # Mock data
    #     target = np.array([0.4, 0.6])
    #     generated = np.array([0.3, 0.7])
    #     mock_np_asarray.side_effect = lambda x: x

    #     # Test KL divergence calculation
    #     kl = self.inference_instance.kl_divergence(target, generated)
    #     self.assertGreaterEqual(kl, 0, "KL divergence should be non-negative.")

    # def test_sample_from_pmf(self):
    #     # Test sampling from a PMF
    #     pmf = np.array([0.1, 0.2, 0.3, 0.4])
    #     samples = self.inference_instance.sample_from_pmf(pmf=pmf, n_shots=100)

    #     # Validate that samples are integers and non-negative
    #     self.assertTrue((samples >= 0).all(), "Samples should contain non-negative integers.")
    #     self.assertEqual(samples.sum(), 100, "Sum of samples should equal the number of shots.")


if __name__ == "__main__":
    unittest.main()
