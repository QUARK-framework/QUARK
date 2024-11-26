import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from modules.applications.qml.generative_modeling.training.QCBM import QCBM


class TestQCBM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.qcbm_instance = QCBM()

    def test_get_requirements(self):
        requirements = self.qcbm_instance.get_requirements()
        expected_requirements = [
            {"name": "numpy", "version": "1.26.4"},
            {"name": "cma", "version": "4.0.0"},
            {"name": "matplotlib", "version": "3.7.5"},
            {"name": "tensorboard", "version": "2.17.0"},
            {"name": "tensorboardX", "version": "2.6.2.2"}
        ]
        self.assertEqual(requirements, expected_requirements)

    def test_get_parameter_options(self):
        parameter_options = self.qcbm_instance.get_parameter_options()
        self.assertIn("population_size", parameter_options)
        self.assertIn("max_evaluations", parameter_options)
        self.assertIn("sigma", parameter_options)
        self.assertIn("pretrained", parameter_options)
        self.assertIn("loss", parameter_options)

    def test_get_default_submodule(self):
        with self.assertRaises(ValueError):
            self.qcbm_instance.get_default_submodule("AnyOption")

    @patch("numpy.random.rand")
    def test_setup_training(self, mock_rand):
        # Mock inputs
        mock_rand.return_value = np.array([0.5, 0.5, 0.5])
        input_data = {
            "backend": "test_backend",
            "n_qubits": 5,
            "store_dir_iter": "./test_dir",
            "n_params": 3
        }
        config = {
            "population_size": 5,
            "max_evaluations": 100,
            "sigma": 0.5,
            "pretrained": "False",
            "loss": "KL"
        }

        x0, options = self.qcbm_instance.setup_training(input_data, config)
        self.assertEqual(x0.shape, (3,))
        self.assertIn("bounds", options)
        self.assertEqual(options["popsize"], 5)

    def test_start_training(self):
        # Mock input data and configuration
        input_data = {
            "backend": "aer_simulator",
            "n_qubits": 4,  # 2^4 = 16 states
            "store_dir_iter": "/tmp/test_qcbm",
            "n_params": 16,
            "generalization_metrics": MagicMock(),
            "n_shots": 1000,
            "histogram_train": np.full(16, 1 / 16),
            "execute_circuit": MagicMock(
                return_value=(np.tile(np.full(16, 1 / 16), (10, 1)), None)
            ),
            "dataset_name": "test_dataset",
        }

        config = {
            "loss": "KL",
            "population_size": 10,
            "max_evaluations": 1000,
            "sigma": 0.5,
            "pretrained": "False",
        }
        try:
            result = self.qcbm_instance.start_training(input_data, config)

            # Validate results
            self.assertIn("best_parameters", result, "Result should include 'best_parameters'.")
            self.assertIn("KL", result, "Result should include 'KL'.")
            self.assertGreater(len(result["best_parameters"]), 0, "Best parameters should not be empty.")
            self.assertGreater(len(result["KL"]), 0, "KL values should not be empty.")
        except ValueError as e:
            # Print PMF and population details for debugging
            print(f"PMF size: {len(input_data['execute_circuit'].return_value[0])}")
            print(f"PMF sum: {np.sum(input_data['execute_circuit'].return_value[0], axis=1)}")
            print(f"Population size: {config['population_size']}")
            raise e

    def test_data_visualization(self):
        import matplotlib.pyplot as plt

        # Mock generalization metrics
        self.qcbm_instance.study_generalization = True
        self.qcbm_instance.generalization_metrics = MagicMock()
        self.qcbm_instance.generalization_metrics.n_shots = 1000

        # Mock writer
        self.qcbm_instance.writer = MagicMock()

        # Define target explicitly
        self.qcbm_instance.target = np.array([0.1] * 16)
        self.qcbm_instance.target[self.qcbm_instance.target == 0] = 1e-8

        # Define `n_qubits` and `n_states_range`
        n_qubits = 4
        self.qcbm_instance.n_states_range = np.arange(2 ** n_qubits)

        self.qcbm_instance.fig, self.qcbm_instance.ax = plt.subplots()

        loss_epoch = np.array([0.1, 0.2, 0.3])
        pmfs_model = np.array([[0.1] * 16])
        pmfs_model /= pmfs_model.sum(axis=1, keepdims=True)
        samples = None

        best_pmf = self.qcbm_instance.data_visualization(loss_epoch, pmfs_model, samples, epoch=1)

        # Validate the results
        self.assertIsNotNone(best_pmf, "Best PMF should not be None.")
        self.qcbm_instance.writer.add_scalar.assert_called()
        self.qcbm_instance.writer.add_figure.assert_called_with('grid_figure', self.qcbm_instance.fig, global_step=1)