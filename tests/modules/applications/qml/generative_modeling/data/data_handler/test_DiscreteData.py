import unittest
from unittest.mock import MagicMock
import numpy as np
from modules.applications.qml.generative_modeling.data.data_handler.DiscreteData import DiscreteData
from modules.applications.qml.generative_modeling.circuits.CircuitCardinality import CircuitCardinality
from modules.applications.qml.generative_modeling.metrics.MetricsGeneralization import MetricsGeneralization


class TestDiscreteData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_handler = DiscreteData()

    def test_initialization(self):
        self.assertEqual(self.data_handler.name, "DiscreteData")
        self.assertEqual(self.data_handler.submodule_options, ["CircuitCardinality"])
        self.assertIsNone(self.data_handler.n_registers)

    def test_get_requirements(self):
        requirements = self.data_handler.get_requirements()
        expected_requirements = [{"name": "numpy", "version": "1.26.4"}]
        self.assertEqual(requirements, expected_requirements)

    def test_get_default_submodule(self):
        submodule = self.data_handler.get_default_submodule("CircuitCardinality")
        self.assertIsInstance(submodule, CircuitCardinality)
        with self.assertRaises(NotImplementedError):
            self.data_handler.get_default_submodule("InvalidSubmodule")

    def test_get_parameter_options(self):
        parameter_options = self.data_handler.get_parameter_options()
        expected_options = {
            "train_size": {
                "values": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                "description": "What percentage of the dataset do you want to use for training?"
            }
        }
        self.assertEqual(parameter_options, expected_options)

    def test_data_load(self):
        gen_mod = {
            "n_qubits": 4,
            "store_dir_iter": "/tmp"
        }
        config = {
            "train_size": 0.5
        }

        application_config = self.data_handler.data_load(gen_mod, config)

        self.assertEqual(application_config["dataset_name"], "Cardinality_Constraint")
        self.assertEqual(application_config["train_size"], 0.5)
        self.assertEqual(application_config["n_qubits"], 4)
        self.assertEqual(application_config["n_registers"], 2)
        self.assertIn("histogram_solution", application_config)
        self.assertIn("histogram_train", application_config)
        self.assertIn("binary_train", application_config)
        self.assertIn("binary_solution", application_config)

        if "generalization_metrics" in application_config:
            self.assertIsInstance(application_config["generalization_metrics"], MetricsGeneralization)

    def test_generalization(self):
        mock_metrics = MagicMock()
        mock_metrics.get_metrics.return_value = {"accuracy": 0.95, "diversity": 0.85}
        self.data_handler.generalization_metrics = mock_metrics

        results, time_taken = self.data_handler.generalization()

        self.assertEqual(results, {"accuracy": 0.95, "diversity": 0.85})
        self.assertGreater(time_taken, 0)

    def test_evaluate(self):
        solution = {
            "best_sample": [10, 20, 30, 40],
            "KL": [0.2, 0.1, 0.15]
        }

        evaluate_dict, time_taken = self.data_handler.evaluate(solution)

        self.assertIn("histogram_generated", evaluate_dict)
        self.assertIn("KL_best", evaluate_dict)
        self.assertGreater(evaluate_dict["KL_best"], 0)
        self.assertGreater(time_taken, 0)
        np.testing.assert_almost_equal(np.sum(evaluate_dict["histogram_generated"]), 1.0, decimal=6)
