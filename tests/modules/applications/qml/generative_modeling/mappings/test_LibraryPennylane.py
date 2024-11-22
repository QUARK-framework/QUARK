import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pennylane as qml
from modules.applications.qml.generative_modeling.mappings.LibraryPennylane import LibraryPennylane
from modules.applications.qml.generative_modeling.training.QCBM import QCBM
from modules.applications.qml.generative_modeling.training.QGAN import QGAN
from modules.applications.qml.generative_modeling.training.Inference import Inference


class TestLibraryPennylane(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.library_instance = LibraryPennylane()

    def test_initialization(self):
        self.assertEqual(self.library_instance.name, "LibraryPennylane")
        self.assertEqual(self.library_instance.submodule_options, ["QCBM", "QGAN", "Inference"])

    def test_get_requirements(self):
        requirements = self.library_instance.get_requirements()
        expected_requirements = [
            {"name": "pennylane", "version": "0.37.0"},
            {"name": "pennylane-lightning", "version": "0.38.0"},
            {"name": "numpy", "version": "1.26.4"},
            {"name": "jax", "version": "0.4.30"},
            {"name": "jaxlib", "version": "0.4.30"}
        ]
        self.assertEqual(requirements, expected_requirements)

    def test_get_parameter_options(self):
        parameter_options = self.library_instance.get_parameter_options()
        expected_options = {
            "backend": {
                "values": ["default.qubit", "default.qubit.jax", "lightning.qubit", "lightning.gpu"],
                "description": "Which device do you want to use?"
            },
            "n_shots": {
                "values": [100, 1000, 10000, 1000000],
                "description": "How many shots do you want use for estimating the PMF of the model?"
            }
        }
        self.assertEqual(parameter_options, expected_options)

    def test_get_default_submodule(self):
        submodule = self.library_instance.get_default_submodule("QCBM")
        self.assertIsInstance(submodule, QCBM)

        submodule = self.library_instance.get_default_submodule("QGAN")
        self.assertIsInstance(submodule, QGAN)

        submodule = self.library_instance.get_default_submodule("Inference")
        self.assertIsInstance(submodule, Inference)

        with self.assertRaises(NotImplementedError):
            self.library_instance.get_default_submodule("InvalidSubmodule")

    def test_sequence_to_circuit(self):
        input_data = {
            "gate_sequence": [
                ["Hadamard", [0]],
                ["CNOT", [0, 1]],
                ["RX", [1]],
                ["RY", [0]],
                ["RXX", [0, 1]]
            ],
            "n_qubits": 2
        }

        output = self.library_instance.sequence_to_circuit(input_data)

        self.assertIn("n_params", output)
        self.assertEqual(output["n_params"], 3)
        self.assertIn("circuit", output)
        self.assertTrue(callable(output["circuit"]))

    def test_select_backend(self):
        # Test default.qubit
        backend = self.library_instance.select_backend("default.qubit", 2)
        self.assertEqual(backend.name, "default.qubit")
        self.assertEqual(len(backend.wires), 2)

        # Test lightning.qubit
        backend = self.library_instance.select_backend("lightning.qubit", 3)
        self.assertEqual(backend.name, "lightning.qubit")
        self.assertEqual(len(backend.wires), 3)

        # Test default.qubit.jax
        backend = self.library_instance.select_backend("default.qubit.jax", 4)
        self.assertEqual(backend.name, "Default qubit (jax) PennyLane plugin")
        self.assertEqual(len(backend.wires), 4)

        # Test invalid backend
        with self.assertRaises(NotImplementedError):
            self.library_instance.select_backend("invalid.backend", 2)



    @patch("pennylane.QNode")
    def test_get_execute_circuit(self, mock_qnode):
        mock_qnode.return_value = lambda x: np.array([0.5, 0.5])  # Mock the qnode

        mock_backend = MagicMock()
        config_dict = {"n_shots": 100}
        circuit = lambda x: x  # Placeholder circuit function

        execute_circuit, _ = self.library_instance.get_execute_circuit(circuit, mock_backend, "default.qubit", config_dict)

        solutions = [np.array([0.1, 0.9]), np.array([0.8, 0.2])]
        pmfs, samples = execute_circuit(solutions)

        self.assertIsInstance(pmfs, np.ndarray)
        self.assertEqual(pmfs.shape, (2, 2))
        self.assertIsInstance(samples, np.ndarray)
        self.assertEqual(samples.shape, (2, 2))
