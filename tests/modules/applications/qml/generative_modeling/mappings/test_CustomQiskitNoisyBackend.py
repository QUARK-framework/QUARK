import unittest
from unittest.mock import ANY, MagicMock, patch

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from modules.applications.qml.generative_modeling.mappings.CustomQiskitNoisyBackend import CustomQiskitNoisyBackend


class TestCustomQiskitNoisyBackend(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.backend_instance = CustomQiskitNoisyBackend()

    def test_initialization(self):
        self.assertEqual(self.backend_instance.name, "CustomQiskitNoisyBackend")
        self.assertEqual(self.backend_instance.submodule_options, ["QCBM", "Inference"])
        self.assertIsNone(self.backend_instance.circuit_transpiled)

    def test_get_requirements(self):
        requirements = self.backend_instance.get_requirements()
        expected_requirements = [
            {"name": "qiskit", "version": "1.3.0"},
            {"name": "qiskit_aer", "version": "0.15.1"},
            {"name": "numpy", "version": "1.26.4"}
        ]
        self.assertEqual(requirements, expected_requirements)

    def test_get_parameter_options(self):
        parameter_options = self.backend_instance.get_parameter_options()
        self.assertIn("backend", parameter_options)
        self.assertIn("simulation_method", parameter_options)
        self.assertIn("n_shots", parameter_options)
        self.assertIn("transpile_optimization_level", parameter_options)
        self.assertIn("noise_configuration", parameter_options)

    def test_get_default_submodule(self):
        submodules = ["QCBM", "Inference"]
        for option in submodules:
            with self.subTest(option=option):
                submodule = self.backend_instance.get_default_submodule(option)
                self.assertIsNotNone(submodule, f"Expected non-None submodule for {option}")
                self.assertIn(type(submodule).__name__, submodules, f"Unexpected submodule type for {option}")

        with self.assertRaises(NotImplementedError):
            self.backend_instance.get_default_submodule("InvalidOption")

    def test_sequence_to_circuit(self):
        input_data = {
            "n_qubits": 3,
            "gate_sequence": [
                ["Hadamard", [0]],
                ["RZ", [1]],
                ["CNOT", [0, 1]],
                ["Measure", [0, 0]]
            ]
        }

        output_data = self.backend_instance.sequence_to_circuit(input_data)

        self.assertIn("circuit", output_data)
        self.assertIsInstance(output_data["circuit"], QuantumCircuit)
        self.assertEqual(output_data["n_params"], 1)  # One parameterized gate in the sequence

    @patch("modules.applications.qml.generative_modeling.mappings.CustomQiskitNoisyBackend.Aer.get_backend")
    def test_select_backend(self, mock_get_backend):
        # Mock the backend and its set_options method
        mock_backend = MagicMock()
        mock_backend.set_options = MagicMock()
        mock_get_backend.return_value = mock_backend

        # Test CPU configuration
        backend = self.backend_instance.select_backend("aer_simulator_cpu", 3)
        self.assertEqual(backend, mock_backend)
        mock_get_backend.assert_called_once_with("aer_simulator")
        mock_backend.set_options.assert_called_once_with(device="CPU")

        # Reset mocks to test GPU configuration
        mock_get_backend.reset_mock()
        mock_backend.reset_mock()

        # Test GPU configuration
        backend = self.backend_instance.select_backend("aer_simulator_gpu", 3)
        self.assertEqual(backend, mock_backend)
        mock_get_backend.assert_called_once_with("aer_simulator")
        mock_backend.set_options.assert_called_once_with(device="GPU")

        # Test invalid configuration
        with self.assertRaises(NotImplementedError):
            self.backend_instance.select_backend("unknown_backend", 3)

    @patch(
        "modules.applications.qml.generative_modeling.mappings.CustomQiskitNoisyBackend.Layout"
    )
    @patch(
        "modules.applications.qml.generative_modeling.mappings.CustomQiskitNoisyBackend.PassManager"
    )
    @patch(
        "modules.applications.qml.generative_modeling.mappings.CustomQiskitNoisyBackend.transpile"
    )
    @patch(
        "modules.applications.qml.generative_modeling.mappings.CustomQiskitNoisyBackend.AerSimulator"
    )
    @patch(
        "modules.applications.qml.generative_modeling.mappings.CustomQiskitNoisyBackend."
        "CustomQiskitNoisyBackend.decompile_noisy_config"
    )
    # pylint: disable=R0917
    def test_get_execute_circuit(self, mock_decompile_noisy_config, mock_aer_simulator,
                                 mock_transpile, mock_pass_manager, mock_layout):
        # Mock Configurations
        mock_backend = MagicMock(spec=AerSimulator)
        mock_decompile_noisy_config.return_value = mock_backend
        mock_pass_manager.return_value.run.return_value = "processed_circuit"

        # Mock Circuit for Transpilation
        mock_transpiled_circuit = MagicMock(spec=QuantumCircuit)
        mock_transpiled_circuit.count_ops.return_value = {"h": 3, "cx": 2}
        mock_transpiled_circuit.assign_parameters = MagicMock()
        mock_transpile.return_value = mock_transpiled_circuit

        # Mock Circuit
        mock_circuit = MagicMock(spec=QuantumCircuit)
        mock_circuit.num_qubits = 3
        mock_circuit.count_ops.return_value = {"h": 3, "cx": 2}

        # Mock Backend Run
        mock_job = MagicMock()
        mock_aer_simulator.return_value.run.return_value = mock_job
        mock_job.result.return_value.get_counts.return_value.int_outcomes.return_value = {0: 10, 1: 20}

        # Config Dictionary
        config_dict = {
            "n_shots": 100,
            "transpile_optimization_level": 2,
            "backend": "aer_simulator_cpu",
            "simulation_method": "statevector",
            "noise_configuration": "No noise",
            "custom_readout_error": 0.01,
            "two_qubit_depolarizing_errors": 0.02,
            "one_qubit_depolarizing_errors": 0.005,
            "qubit_layout": "linear",
        }

        # Call the method
        execute_circuit, circuit_transpiled = self.backend_instance.get_execute_circuit(
            circuit=mock_circuit,
            backend=mock_backend,
            config="aer_simulator_cpu",
            config_dict=config_dict,
        )

        # Assertions
        self.assertEqual(circuit_transpiled, mock_transpiled_circuit)
        self.assertTrue(callable(execute_circuit))

        # Mock Solutions
        solutions = [{"param_0": 0.5}, {"param_0": 1.0}]
        pmfs, samples = execute_circuit(solutions)

        # Assertions on returned values
        self.assertIsInstance(pmfs, np.ndarray)
        self.assertIsInstance(samples, np.ndarray)
        self.assertEqual(pmfs.shape[0], len(solutions))
        self.assertEqual(samples.shape[0], len(solutions))

        # Check calls to mocks
        mock_decompile_noisy_config.assert_called_once_with(config_dict, 3)
        mock_pass_manager.return_value.run.assert_called_once_with(mock_circuit)
        mock_transpile.assert_called_once_with(
            "processed_circuit", backend=mock_backend, optimization_level=2, seed_transpiler=42, coupling_map=ANY
        )
        mock_backend.run.assert_called_once()

    @patch(
        "modules.applications.qml.generative_modeling.mappings.CustomQiskitNoisyBackend."
        "CustomQiskitNoisyBackend.build_noise_model"
    )
    @patch(
        "modules.applications.qml.generative_modeling.mappings.CustomQiskitNoisyBackend."
        "CustomQiskitNoisyBackend.get_coupling_map"
    )
    @patch(
        "modules.applications.qml.generative_modeling.mappings.CustomQiskitNoisyBackend."
        "Aer.get_backend"
    )
    @patch(
        "modules.applications.qml.generative_modeling.mappings.CustomQiskitNoisyBackend."
        "CustomQiskitNoisyBackend.log_backend_options"
    )
    def test_decompile_noisy_config(self, mock_log_backend_options, mock_get_backend,
                                    mock_get_coupling_map, mock_build_noise_model):
        # Mock simulation method and device
        simulation_method = "statevector"
        device = "CPU"

        # Mock noise model
        mock_noise_model = MagicMock(spec=NoiseModel)
        mock_build_noise_model.return_value = mock_noise_model

        # Mock coupling map
        mock_coupling_map = MagicMock()
        mock_get_coupling_map.return_value = mock_coupling_map

        # Mock Aer backend
        mock_backend = MagicMock(spec=AerSimulator)
        mock_backend.set_options = MagicMock()
        mock_backend.name = "aer_simulator_statevector"
        mock_get_backend.return_value = mock_backend

        # Test default AerSimulator configuration
        config_dict = {
            "backend": "aer_simulator_cpu",
            "simulation_method": "statevector",
            "noise_configuration": "No noise"
        }
        num_qubits = 4

        backend = self.backend_instance.decompile_noisy_config(config_dict, num_qubits)

        self.assertEqual(backend.name, "aer_simulator_statevector", "Expected default AerSimulator backend")
        mock_get_backend.assert_called_once_with("aer_simulator")
        mock_backend.set_options.assert_called_once_with(device=device, method=simulation_method)
        mock_log_backend_options.assert_called_once_with(mock_backend)

        # Reset mocks for the next test case
        mock_get_backend.reset_mock()
        mock_backend.set_options.reset_mock()
        mock_log_backend_options.reset_mock()

        # Test custom noise configuration
        config_dict["noise_configuration"] = "Custom configurations"
        backend = self.backend_instance.decompile_noisy_config(config_dict, num_qubits)

        # Assertions for custom backend
        self.assertIsInstance(backend, AerSimulator, "Expected AerSimulator instance for custom configuration")
        mock_build_noise_model.assert_called_once_with(config_dict)
        mock_get_coupling_map.assert_called_once_with(config_dict, num_qubits)

    def test_build_noise_model(self):
        config_dict = {
            "custom_readout_error": 0.01,
            "two_qubit_depolarizing_errors": 0.02,
            "one_qubit_depolarizing_errors": 0.005
        }

        noise_model = self.backend_instance.build_noise_model(config_dict)
        self.assertIsInstance(noise_model, NoiseModel)

    def test_get_custom_config(self):
        config_dict = {
            "custom_readout_error": 0.01,
            "two_qubit_depolarizing_errors": 0.02,
            "one_qubit_depolarizing_errors": 0.005,
            "qubit_layout": "linear",
            "backend": "aer_simulator_cpu"
        }

        backend = self.backend_instance.get_custom_config(config_dict, num_qubits=3)
        self.assertIsInstance(backend, AerSimulator)

    def test_get_simulation_method_and_device(self):
        # Test statevector configuration
        simulation_method, device = self.backend_instance.get_simulation_method_and_device("CPU", "statevector")
        self.assertEqual(simulation_method, "statevector")
        self.assertEqual(device, "CPU")

        # Test cpu_mps configuration, which forces the device to CPU
        simulation_method, device = self.backend_instance.get_simulation_method_and_device("GPU", "cpu_mps")
        self.assertEqual(simulation_method, "matrix_product_state")
        self.assertEqual(device, "CPU", "Device should be forced to CPU for cpu_mps simulation.")

        # Test density_matrix configuration
        simulation_method, device = self.backend_instance.get_simulation_method_and_device("GPU", "density_matrix")
        self.assertEqual(simulation_method, "density_matrix")
        self.assertEqual(device, "GPU")

        # Test default simulation method (automatic)
        simulation_method, device = self.backend_instance.get_simulation_method_and_device("GPU", "unknown_method")
        self.assertEqual(simulation_method, "automatic", "Expected 'automatic' for unknown simulation methods.")
        self.assertEqual(device, "GPU", "Device should remain unchanged for unknown simulation methods.")

    def test_get_transpile_routine(self):
        self.assertEqual(self.backend_instance.get_transpile_routine(2), 2)
        self.assertEqual(self.backend_instance.get_transpile_routine(5), 1)  # Invalid config defaults to 1

    @patch("modules.applications.qml.generative_modeling.mappings.CustomQiskitNoisyBackend.noise.depolarizing_error")
    def test_add_quantum_errors(self, mock_depolarizing_error):
        # Mock noise model and depolarizing errors
        mock_noise_model = MagicMock(spec=NoiseModel)
        mock_two_qubit_error = MagicMock()
        mock_one_qubit_error = MagicMock()
        mock_depolarizing_error.side_effect = [mock_two_qubit_error, mock_one_qubit_error]

        config_dict = {
            "two_qubit_depolarizing_errors": 0.02,
            "one_qubit_depolarizing_errors": 0.005,
        }

        self.backend_instance.add_quantum_errors(mock_noise_model, config_dict)

        # Assertions for two-qubit errors
        mock_depolarizing_error.assert_any_call(0.02, 2)
        for gate in ['cx', 'ecr', 'rxx']:
            mock_noise_model.add_all_qubit_quantum_error.assert_any_call(mock_two_qubit_error, gate)

        # Assertions for one-qubit errors
        mock_depolarizing_error.assert_any_call(0.005, 1)
        for gate in ['sx', 'x', 'rx', 'ry', 'rz', 'h', 's']:
            mock_noise_model.add_all_qubit_quantum_error.assert_any_call(mock_one_qubit_error, gate)

    def test_get_coupling_map(self):
        # Test linear coupling map
        config_dict = {"qubit_layout": "linear"}
        coupling_map = self.backend_instance.get_coupling_map(config_dict, num_qubits=4)
        self.assertIsNotNone(coupling_map)
        self.assertIsInstance(coupling_map, CouplingMap, "Expected a CouplingMap instance for linear layout.")
        self.assertEqual(coupling_map.size(), 4, "Coupling map size should match the number of qubits.")

        # Test circular coupling map
        config_dict = {"qubit_layout": "circle"}
        coupling_map = self.backend_instance.get_coupling_map(config_dict, num_qubits=4)
        self.assertIsInstance(coupling_map, CouplingMap, "Expected a CouplingMap instance for circular layout.")

        # Test fully connected coupling map
        config_dict = {"qubit_layout": "fully_connected"}
        coupling_map = self.backend_instance.get_coupling_map(config_dict, num_qubits=4)
        self.assertIsInstance(coupling_map, CouplingMap, "Expected a CouplingMap instance for fully connected layout.")

        # Test no specified coupling map
        config_dict = {"qubit_layout": None}
        coupling_map = self.backend_instance.get_coupling_map(config_dict, num_qubits=4)
        self.assertIsNone(coupling_map)

        # Test unknown qubit layout
        config_dict = {"qubit_layout": "unknown_layout"}
        with self.assertRaises(ValueError):
            self.backend_instance.get_coupling_map(config_dict, num_qubits=4)
