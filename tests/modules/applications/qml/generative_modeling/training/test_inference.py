import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.modules.applications.qml.generative_modeling.training.inference import Inference


class TestInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.inference_instance = Inference()
        cls.mock_parameters = np.array([0.5, 0.2, 0.3])
        cls.config = {"pretrained": "mock_pretrained_file.npy"}
        cls.input_data = {
            "n_qubits": 3,
            "histogram_train": np.array([0.1, 0.2, 0.3, 0.4]),
            "execute_circuit": MagicMock(),
            "n_shots": 1000,
        }

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

    @patch("numpy.load")
    def test_start_training(self, mock_np_load):
        # Mock np.load to return mock parameters
        mock_np_load.return_value = self.mock_parameters

        # Update the execute_circuit mock to return a PMF with the correct size
        n_states = 2 ** self.input_data["n_qubits"]
        pmf_mock = np.full(n_states, 1 / n_states)
        self.input_data["execute_circuit"].return_value = ([pmf_mock], None)

        result = self.inference_instance.start_training(self.input_data, self.config)

        # Validate the returned dictionary keys
        self.assertIn("best_parameter", result, "'best_parameter' key is missing from the result.")
        self.assertIn("inference", result, "'inference' key is missing from the result.")
        self.assertIn("KL", result, "'KL' key is missing from the result.")
        self.assertIn("best_sample", result, "'best_sample' key is missing from the result.")

        self.assertTrue(result["inference"], "The 'inference' flag should be True.")

       # Extract KL divergence
        kl_values = result["KL"]
        self.assertIsInstance(kl_values, list, "KL divergence values should be returned as a list.")

        self.assertTrue((result["best_sample"] >= 0).all(), "Best sample should contain non-negative integers.")

        best_parameter = result["best_parameter"]
        np.testing.assert_array_equal(
            best_parameter,
            self.mock_parameters,
            "Best parameter does not match the expected parameters.")
