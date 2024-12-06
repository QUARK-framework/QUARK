import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from torch.utils.data import DataLoader
from modules.applications.qml.generative_modeling.training.QGAN import QGAN, Discriminator, QuantumGenerator


class TestQGAN(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.qgan_instance = QGAN()
        cls.input_length = 4
        cls.discriminator = Discriminator(input_length=cls.input_length)
        cls.input_data = {
            "n_qubits": 4,
            "n_registers": 2,
            "n_shots": 1000,
            "train_size": 0.8,
            "execute_circuit": MagicMock(),
            "store_dir_iter": "/tmp/qgan",
            "binary_train": np.random.randint(2, size=(100, 4)),
            "dataset_name": "Test_Dataset",
            "histogram_train": np.random.random(16),
            "n_params": 10
        }
        cls.input_data["histogram_train"] /= cls.input_data["histogram_train"].sum()  # Normalize histogram
        cls.config = {
            "epochs": 2,
            "batch_size": 10,
            "learning_rate_generator": 0.1,
            "learning_rate_discriminator": 0.05,
            "device": "cpu",
            "loss": "KL"
        }

    def test_get_requirements(self):
        requirements = self.qgan_instance.get_requirements()
        expected_requirements = [
            {"name": "numpy", "version": "1.26.4"},
            {"name": "torch", "version": "2.2.0"},
            {"name": "matplotlib", "version": "3.7.5"},
            {"name": "tensorboard", "version": "2.17.0"},
            {"name": "tensorboardX", "version": "2.6.2.2"}
        ]
        self.assertEqual(requirements, expected_requirements)

    def test_get_parameter_options(self):
        parameter_options = self.qgan_instance.get_parameter_options()
        self.assertIn("epochs", parameter_options)
        self.assertIn("batch_size", parameter_options)
        self.assertIn("learning_rate_generator", parameter_options)

    def test_get_default_submodule_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.qgan_instance.get_default_submodule("option")

    def test_setup_training(self):
        self.qgan_instance.setup_training(self.input_data, self.config)

        total_samples = self.qgan_instance.bins_train.shape[0]
        batch_size = self.config["batch_size"]
        expected_batches = total_samples // batch_size

        actual_batches = len(self.qgan_instance.dataloader)

        self.assertEqual(
            actual_batches,
            expected_batches,
            f"Expected {expected_batches} batches, but got {actual_batches}."
        )
        self.assertEqual(self.qgan_instance.n_qubits, 4)
        self.assertEqual(self.qgan_instance.device, "cpu")
        self.assertEqual(self.qgan_instance.batch_size, 10)
        self.assertIsInstance(self.qgan_instance.dataloader, DataLoader)

        drop_last = self.qgan_instance.dataloader.drop_last
        self.assertTrue(drop_last, "drop_last should be True to avoid partial batches.")

    def test_start_training(self):
        # Mock the execute_circuit to return expected values
        self.input_data["execute_circuit"] = MagicMock(
            return_value=(np.random.rand(1, 16), None)
        )

        result = self.qgan_instance.start_training(self.input_data, self.config)

        self.assertIn("best_parameter", result, "The result should contain 'best_parameter'.")
        self.assertIn("best_sample", result, "The result should contain 'best_sample'.")
        self.assertIn("KL", result, "The result should contain 'KL'.")
        self.assertIn("generator_loss", result, "The result should contain 'generator_loss'.")
        self.assertIn("discriminator_loss", result, "The result should contain 'discriminator_loss'.")

    def test_discriminator_forward(self):
        input_tensor = torch.rand(10, self.input_length)
        output = self.discriminator(input_tensor)
        self.assertEqual(output.shape, (10, 1), "The output shape should be (10, 1).")

    def test_discriminator_weights_init(self):
        discriminator = Discriminator(input_length=self.input_length)
        # Apply the weights initialization
        discriminator.apply(Discriminator.weights_init)

        for layer in discriminator.children():
            if isinstance(layer, torch.nn.Linear):
                weights = layer.weight.data
                bias = layer.bias.data

                # Check if weights follow Xavier initialization
                self.assertTrue(weights.mean().item() != 0, "Weights mean should not be zero after Xavier init.")

                # Check if biases are initialized to 1
                self.assertTrue(torch.allclose(bias, torch.tensor(1.0)), "Biases should be initialized to 1.")

    def test_quantum_generator_execute(self):
        execute_circuit_mock = MagicMock(return_value=(np.random.random(16), None))

        # Initialize the QuantumGenerator
        generator = QuantumGenerator(n_qubits=4, execute_circuit=execute_circuit_mock, batch_size=10)

        # Use n_shots equal to batch_size
        n_shots = 100
        params = np.random.random(10)

        samples, pdfs = generator.execute(params, n_shots=n_shots)

        self.assertEqual(pdfs.shape, (16,), "Expected PMF size to match the number of qubits (2^n_qubits).")
        self.assertEqual(samples.shape[0], n_shots, f"Expected number of samples to match n_shots ({n_shots}).")

    def test_quantum_generator_compute_gradient(self):
        generator = QuantumGenerator(
            n_qubits=self.input_data["n_qubits"],
            execute_circuit=self.input_data["execute_circuit"],
            batch_size=self.config["batch_size"]
        )

        generator.execute = MagicMock(
            return_value=(
                torch.rand(self.config["batch_size"], self.input_data["n_qubits"]),
                torch.rand(2 ** self.input_data["n_qubits"])
            )
        )

        discriminator = Discriminator(input_length=4)
        criterion = torch.nn.BCELoss()
        label = torch.ones(self.config["batch_size"])

        # Compute gradients
        params = np.random.rand(16)
        gradients = generator.compute_gradient(
            params=params,
            discriminator=discriminator,
            criterion=criterion,
            label=label,
            device="cpu"
        )

        # Assertions
        self.assertEqual(len(gradients), len(params), "Gradient size should match number of parameters.")
        self.assertTrue(np.all(np.isfinite(gradients)), "All gradients should be finite.")
