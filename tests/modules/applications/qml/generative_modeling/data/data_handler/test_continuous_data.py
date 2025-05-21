import unittest
from unittest.mock import patch

import numpy as np

from modules.applications.qml.generative_modeling.data.data_handler.continuous_data import ContinuousData


class TestContinuousData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_handler = ContinuousData()

    def test_initialization(self):
        self.assertEqual(self.data_handler.name, "ContinuousData")
        self.assertEqual(self.data_handler.submodule_options, ["PIT", "MinMax"])

    def test_get_requirements(self):
        requirements = self.data_handler.get_requirements()
        expected_requirements = [{"name": "numpy", "version": "1.26.4"}]
        self.assertEqual(requirements, expected_requirements)

    def test_get_parameter_options(self):
        parameter_options = self.data_handler.get_parameter_options()
        expected_options = {
            "data_set": {
                "values": ["X_2D", "O_2D", "MG_2D", "Stocks_2D"],
                "description": "Which dataset do you want to use?"
            },
            "train_size": {
                "values": [0.1, 0.3, 0.5, 0.7, 1.0],
                "description": "What percentage of the dataset do you want to use for training?"
            }
        }
        self.assertEqual(parameter_options, expected_options)

    def test_get_default_submodule(self):
        # Test MinMax
        submodule = self.data_handler.get_default_submodule("MinMax")
        self.assertIsNotNone(submodule)
        self.assertEqual(self.data_handler.transformation.__class__.__name__, "MinMax")

        # Test PIT
        submodule = self.data_handler.get_default_submodule("PIT")
        self.assertIsNotNone(submodule)
        self.assertEqual(self.data_handler.transformation.__class__.__name__, "PIT")

        # Test invalid submodule
        with self.assertRaises(NotImplementedError):
            self.data_handler.get_default_submodule("InvalidSubmodule")

    @patch(
        "modules.applications.qml.generative_modeling.data.data_handler.continuous_data.pkg_resources.resource_filename"
    )
    @patch(
        "modules.applications.qml.generative_modeling.data.data_handler.continuous_data.np.load"
    )
    def test_data_load(self, mock_np_load, mock_resource_filename):
        mock_resource_filename.return_value = "/mock/path/X_2D.npy"
        mock_np_load.return_value = np.array([[1, 2], [3, 4]])

        gen_mod = {
            "n_qubits": 2,
            "store_dir_iter": "/tmp"
        }
        config = {
            "data_set": "X_2D",
            "train_size": 0.5
        }

        result = self.data_handler.data_load(gen_mod, config)

        mock_resource_filename.assert_called_once_with(
            'modules.applications.qml.generative_modeling.data',
            "X_2D.npy"
        )
        mock_np_load.assert_called_once_with("/mock/path/X_2D.npy")

        self.assertEqual(result["dataset_name"], "X_2D")
        self.assertEqual(result["n_qubits"], 2)
        self.assertEqual(result["train_size"], 0.5)
        self.assertEqual(result["store_dir_iter"], "/tmp")
        np.testing.assert_array_equal(result["dataset"], np.array([[1, 2], [3, 4]]))

    def test_evaluate(self):
        solution = {
            "histogram_generated_original": np.array([0.1, 0.2, 0.3, 0.4]),
            "histogram_train_original": np.array([0.25, 0.25, 0.25, 0.25])
        }
        kl_divergence, time_taken = self.data_handler.evaluate(solution)

        self.assertGreater(kl_divergence, 0)
        self.assertGreater(time_taken, 0)

    def test_kl_divergence(self):
        target = np.array([0.4, 0.3, 0.2, 0.1])
        generated = np.array([0.3, 0.3, 0.3, 0.1])
        kl_divergence = self.data_handler.kl_divergence(target, generated)

        # Expected KL divergence calculation
        expected_kl_divergence = np.sum(target * np.log(target / generated))
        self.assertAlmostEqual(kl_divergence, expected_kl_divergence, places=6)

    def test_evaluate_with_zeros(self):
        # Test handling of zeros in histograms
        solution = {
            "histogram_generated_original": np.array([[0.2, 0.0], [0.0, 0.4]]),
            "histogram_train_original": np.array([[0.25, 0.0], [0.0, 0.25]])
        }

        generated = np.array([[0.2, 1e-8], [1e-8, 0.4]])
        target = np.array([[0.25, 1e-8], [1e-8, 0.25]])
        expected_kl = np.sum(target.ravel() * np.log(target.ravel() / generated.ravel()))

        kl_divergence, _ = self.data_handler.evaluate(solution)
        self.assertAlmostEqual(kl_divergence, expected_kl, places=6, msg="KL divergence with zeros is incorrect.")
