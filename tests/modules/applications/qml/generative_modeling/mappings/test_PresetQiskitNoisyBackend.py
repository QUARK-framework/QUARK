import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pickle
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from modules.applications.qml.generative_modeling.mappings.PresetQiskitNoisyBackend import PresetQiskitNoisyBackend


class TestPresetQiskitNoisyBackend(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.backend_instance = PresetQiskitNoisyBackend()
        with open("tests/modules/applications/qml/generative_modeling/mappings/test_circuit.pkl", "rb") as file:
            cls.circuit = pickle.load(file)

    def test_initialization(self):
        self.assertEqual(self.backend_instance.name, "PresetQiskitNoisyBackend")
        self.assertEqual(self.backend_instance.submodule_options, ["QCBM", "Inference"])

    def test_get_requirements(self):
        requirements = self.backend_instance.get_requirements()
        expected_requirements = [
            {"name": "qiskit", "version": "1.3.0"},
            {"name": "qiskit_ibm_runtime", "version": "0.33.2"},
            {"name": "qiskit_aer", "version": "0.15.1"},
            {"name": "numpy", "version": "1.26.4"}
        ]
        self.assertEqual(requirements, expected_requirements)

    def test_get_default_submodule(self):
        from modules.applications.qml.generative_modeling.training.QCBM import QCBM
        from modules.applications.qml.generative_modeling.training.Inference import Inference

        submodule = self.backend_instance.get_default_submodule("QCBM")
        self.assertIsInstance(submodule, QCBM)

        submodule = self.backend_instance.get_default_submodule("Inference")
        self.assertIsInstance(submodule, Inference)

        with self.assertRaises(NotImplementedError):
            self.backend_instance.get_default_submodule("InvalidSubmodule")

    def test_sequence_to_circuit(self):
        input_data = {
            "n_qubits": 2,
            "gate_sequence": [
                ["Hadamard", [0]],
                ["RX", [1]],
                ["CNOT", [0, 1]],
                ["Measure", [0, 0]],
                ["Measure", [1, 1]],
            ]
        }

        output = self.backend_instance.sequence_to_circuit(input_data)
        self.assertIn("circuit", output)
        self.assertIsInstance(output["circuit"], QuantumCircuit)
        self.assertEqual(output["n_params"], 1)  # Only one RX gate requires a parameter

    @patch("qiskit_aer.Aer.get_backend")
    def test_select_backend(self, mock_get_backend):
        mock_backend = MagicMock()
        mock_get_backend.return_value = mock_backend

        backend = self.backend_instance.select_backend("aer_simulator_cpu", 2)
        mock_get_backend.assert_called_with("aer_simulator")
        mock_backend.set_options.assert_called_with(device="CPU")
        self.assertIs(backend, mock_backend)

        backend = self.backend_instance.select_backend("aer_simulator_gpu", 3)
        mock_get_backend.assert_called_with("aer_simulator")
        mock_backend.set_options.assert_called_with(device="GPU")
        self.assertIs(backend, mock_backend)

        with self.assertRaises(NotImplementedError):
            self.backend_instance.select_backend("invalid_backend", 2)

    @patch("qiskit.transpile")
    @patch("qiskit_aer.AerSimulator.run")
    def test_get_execute_circuit(self, mock_run, mock_transpile):

        # Mock transpile function
        mock_transpiled_circuit = self.circuit.copy()
        mock_transpile.return_value = mock_transpiled_circuit

        # Mock backend behavior
        mock_backend = MagicMock(spec=AerSimulator)
        mock_result = MagicMock()
        mock_result.get_counts.return_value.int_outcomes.return_value = {0: 100, 1: 50}
        mock_run.return_value.result.return_value = mock_result

        # Configuration dictionary
        config_dict = {
            "backend": "aer_simulator_cpu",
            "n_shots": 150,
            "simulation_method": "statevector",
            "noise_configuration": "No noise",
            "transpile_optimization_level": 1
        }

        execute_circuit, transpiled_circuit = self.backend_instance.get_execute_circuit(
            self.circuit, mock_backend, "aer_simulator_cpu", config_dict
        )

        self.assertEqual(
            transpiled_circuit, mock_transpiled_circuit,
            "The transpiled circuit does not match the expected circuit."
        )

        # Define parameter values for solutions
        param_0, param_1 = list(self.circuit.parameters)
        solutions = [{param_0: 1.0, param_1: 2.0}]
        pmfs, samples = execute_circuit(solutions)

        self.assertIsInstance(pmfs, np.ndarray)
        self.assertIsInstance(samples, np.ndarray)

    def test_get_simulation_method_and_device(self):
        # Test valid configuration: statevector
        method, device = self.backend_instance.get_simulation_method_and_device("GPU", "statevector")
        self.assertEqual(method, "statevector")
        self.assertEqual(device, "GPU")

        # Test valid configuration: cpu_mps (forces device to CPU)
        method, device = self.backend_instance.get_simulation_method_and_device("GPU", "cpu_mps")
        self.assertEqual(method, "matrix_product_state")
        self.assertEqual(device, "CPU")

        # Test valid configuration: density_matrix
        method, device = self.backend_instance.get_simulation_method_and_device("GPU", "density_matrix")
        self.assertEqual(method, "density_matrix")
        self.assertEqual(device, "GPU")

        # Test default configuration for unknown simulation method
        method, device = self.backend_instance.get_simulation_method_and_device("GPU", "unknown_method")
        self.assertEqual(method, "automatic")
        self.assertEqual(device, "GPU")

        # Test default device behavior with CPU
        method, device = self.backend_instance.get_simulation_method_and_device("CPU", "unknown_method")
        self.assertEqual(method, "automatic")
        self.assertEqual(device, "CPU")

    def test_get_transpile_routine(self):
        self.assertEqual(self.backend_instance.get_transpile_routine(2), 2)
        self.assertEqual(self.backend_instance.get_transpile_routine(5), 1)  # Invalid option defaults to 1

    @patch("qiskit_aer.Aer.get_backend")
    @patch("modules.applications.qml.generative_modeling.mappings.PresetQiskitNoisyBackend.PresetQiskitNoisyBackend.get_FakeBackend")
    def test_decompile_noisy_config(self, mock_get_fake_backend, mock_get_backend):
        mock_backend = MagicMock(spec=AerSimulator)
        mock_get_backend.return_value = mock_backend
        mock_get_fake_backend.return_value = mock_backend

        # Test No Noise
        config_dict = {
            "backend": "aer_simulator_gpu",
            "simulation_method": "statevector",
            "noise_configuration": "No noise"
        }
        backend = self.backend_instance.decompile_noisy_config(config_dict, 2)
        mock_get_backend.assert_called_once_with("aer_simulator")
        mock_backend.set_options.assert_any_call(device="GPU")
        mock_backend.set_options.assert_any_call(method="statevector")
        self.assertEqual(backend, mock_backend)

        # Test Fake Backend
        config_dict["noise_configuration"] = "fake_backend"
        backend = self.backend_instance.decompile_noisy_config(config_dict, 2)
        mock_get_fake_backend.assert_called_once_with("fake_backend", 2)
        self.assertEqual(backend, mock_backend)

        # Test Invalid Noise Configuration
        config_dict["noise_configuration"] = "invalid_noise"
        with self.assertRaises(ValueError):
            self.backend_instance.decompile_noisy_config(config_dict, 2)

    @patch("qiskit_aer.Aer.get_backend")
    def test_select_backend_configuration(self, mock_get_backend):
        mock_backend = MagicMock(spec=AerSimulator)
        mock_get_backend.return_value = mock_backend

        # Test No Noise
        backend = self.backend_instance.select_backend_configuration("No noise", 3)
        mock_get_backend.assert_called_once_with("aer_simulator")
        self.assertEqual(backend, mock_backend)

        # Test Fake Backend
        with patch.object(self.backend_instance, "get_FakeBackend", return_value=mock_backend) as mock_fake_backend:
            backend = self.backend_instance.select_backend_configuration("fake_backend", 3)
            mock_fake_backend.assert_called_once_with("fake_backend", 3)
            self.assertEqual(backend, mock_backend)

        # Test Invalid Configuration
        with self.assertRaises(ValueError):
            self.backend_instance.select_backend_configuration("invalid_noise", 3)

    def test_configure_backend(self):
        mock_backend = MagicMock()
        self.backend_instance.configure_backend(mock_backend, "GPU", "statevector")
        mock_backend.set_options.assert_any_call(device="GPU")
        mock_backend.set_options.assert_any_call(method="statevector")

    @patch("logging.info")
    def test_log_backend_info(self, mock_logging):
        mock_backend = MagicMock()
        mock_backend.configuration.return_value = {"dummy": "config"}
        mock_backend.options.method = "statevector"
        self.backend_instance.log_backend_info(mock_backend)
        mock_logging.assert_any_call("Backend configuration: {'dummy': 'config'}")
        mock_logging.assert_any_call("Simulation method: statevector")

    @patch("qiskit_aer.noise.NoiseModel.from_backend")
    @patch("qiskit_aer.AerSimulator.from_backend")
    @patch("qiskit_aer.Aer.get_backend")
    @patch("modules.applications.qml.generative_modeling.mappings.PresetQiskitNoisyBackend.FakeProviderForBackendV2")
    def test_get_FakeBackend(self, mock_provider, mock_aer_get_backend, mock_simulator_from_backend, mock_noise_model):
        mock_backend = MagicMock()
        mock_backend.num_qubits = 5
        mock_backend.name = "fake_backend_name"
        mock_provider.return_value.get_backend.return_value = mock_backend
        mock_noise_model.return_value = MagicMock()
        mock_aer_get_backend.return_value = MagicMock()

        with self.subTest("Retrieve backend successfully"):
            backend = self.backend_instance.get_FakeBackend("fake_backend_name", 4)
            mock_provider.return_value.get_backend.assert_called_once_with("fake_backend_name")
            mock_noise_model.assert_called_once_with(mock_backend)
            mock_simulator_from_backend.assert_called_once_with(mock_backend)
            self.assertEqual(backend, mock_simulator_from_backend.return_value)
