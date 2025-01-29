import unittest
from unittest.mock import patch, MagicMock
from qiskit import QuantumCircuit
import numpy as np
from qiskit_aer import AerSimulator

from quark.modules.applications.qml.generative_modeling.mappings.LibraryQiskit import LibraryQiskit
from quark.modules.applications.qml.generative_modeling.training.QCBM import QCBM
from quark.modules.applications.qml.generative_modeling.training.QGAN import QGAN
from quark.modules.applications.qml.generative_modeling.training.Inference import Inference


class TestLibraryQiskit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.library_instance = LibraryQiskit()

    def test_initialization(self):
        self.assertEqual(self.library_instance.name, "LibraryQiskit")
        self.assertEqual(self.library_instance.submodule_options, ["QCBM", "QGAN", "Inference"])

    def test_get_requirements(self):
        requirements = self.library_instance.get_requirements()
        expected_requirements = [
            {"name": "qiskit", "version": "1.3.0"},
            {"name": "numpy", "version": "1.26.4"}
        ]
        self.assertEqual(requirements, expected_requirements)

    def test_get_parameter_options(self):
        parameter_options = self.library_instance.get_parameter_options()
        expected_options = {
            "backend": {
                "values": ["aer_statevector_simulator_gpu", "aer_statevector_simulator_cpu",
                           "cusvaer_simulator (only available in cuQuantum appliance)", "aer_simulator_gpu",
                           "aer_simulator_cpu", "ionQ_Harmony", "Amazon_SV1", "ibm_brisbane IBM Quantum Platform"],
                "description": "Which backend do you want to use? (aer_statevector_simulator uses the measurement "
                               "probability vector, the others are shot based)"
            },
            "n_shots": {
                "values": [100, 1000, 10000, 1000000],
                "description": "How many shots do you want use for estimating the PMF of the model? "
                               "(If the aer_statevector_simulator selected, only relevant for studying generalization)"
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
            "n_qubits": 2,
            "gate_sequence": [
                ["Hadamard", [0]],
                ["CNOT", [0, 1]],
                ["RX", [0]],
                ["RY", [1]],
                ["RXX", [0, 1]]
            ]
        }

        output = self.library_instance.sequence_to_circuit(input_data)

        self.assertIn("circuit", output)
        self.assertIsInstance(output["circuit"], QuantumCircuit)
        self.assertIn("n_params", output)
        self.assertEqual(output["n_params"], 3)  # RX, RY, RXX need 3 parameters

    # These tests are currently commented out because implementing test cases for the
    # cusvaer simulator is challenging due to the complexity of mocking certain
    # behaviors of the `cusvaer`-enabled backend. We plan to implement these tests
    # in the future once we have resolved these issues.
    # @patch("modules.applications.qml.generative_modeling.mappings.LibraryQiskit.select_backend.cusvaer")
    # @patch("qiskit_aer.Aer.get_backend")
    # def test_cusvaer_simulator(self, mock_aer_simulator, mock_cusvaer):
    #     mock_backend = MagicMock()
    #     mock_aer_simulator.return_value = mock_backend

    #     backend = self.library_instance.select_backend(
    #         "cusvaer_simulator (only available in cuQuantum appliance)", 5
    #     )
    #     self.assertEqual(backend, mock_backend)
    #     mock_aer_simulator.assert_called_once_with(
    #         method="statevector",
    #         device="GPU",
    #         cusvaer_enable=True,
    #         noise_model=None,
    #         cusvaer_p2p_device_bits=3,
    #         cusvaer_comm_plugin_type=mock_cusvaer.CommPluginType.MPI_AUTO,
    #         cusvaer_comm_plugin_soname="libmpi.so",
    #     )

    @patch("qiskit_aer.Aer.get_backend")
    def test_aer_simulator_gpu(self, mock_get_backend):
        mock_backend = MagicMock()
        mock_get_backend.return_value = mock_backend

        backend = self.library_instance.select_backend("aer_simulator_gpu", 4)
        mock_get_backend.assert_called_once_with("aer_simulator")
        mock_backend.set_options.assert_called_once_with(device="GPU")
        self.assertEqual(backend, mock_backend)

    @patch("qiskit_aer.Aer.get_backend")
    def test_aer_simulator_cpu(self, mock_get_backend):
        mock_backend = MagicMock()
        mock_get_backend.return_value = mock_backend

        backend = self.library_instance.select_backend("aer_simulator_cpu", 4)
        mock_get_backend.assert_called_once_with("aer_simulator")
        mock_backend.set_options.assert_called_once_with(device="CPU")
        self.assertEqual(backend, mock_backend)

    @patch("qiskit_aer.Aer.get_backend")
    def test_aer_statevector_simulator_gpu(self, mock_get_backend):
        mock_backend = MagicMock()
        mock_get_backend.return_value = mock_backend

        backend = self.library_instance.select_backend("aer_statevector_simulator_gpu", 4)
        mock_get_backend.assert_called_once_with("statevector_simulator")
        mock_backend.set_options.assert_called_once_with(device="GPU")
        self.assertEqual(backend, mock_backend)

    @patch("qiskit_aer.Aer.get_backend")
    def test_aer_statevector_simulator_cpu(self, mock_get_backend):
        mock_backend = MagicMock()
        mock_get_backend.return_value = mock_backend

        backend = self.library_instance.select_backend("aer_statevector_simulator_cpu", 4)
        mock_get_backend.assert_called_once_with("statevector_simulator")
        mock_backend.set_options.assert_called_once_with(device="CPU")
        self.assertEqual(backend, mock_backend)

    # The following tests are commented out because:
    # - The `AWSBraketBackend` and `AWSBraketProvider` are complex to mock in the current setup.
    # - Additional setup or dependency resolution is required for testing with AWS Braket devices (e.g., SV1 or IonQ Harmony).
    # def test_amazon_sv1(self):
    #     from qiskit_braket_provider import AWSBraketBackend, AWSBraketProvider
    #     from modules.devices.braket.SV1 import SV1

    #     # Create a mock device wrapper and backend
    #     device_wrapper = SV1("SV1", "arn:aws:braket:::device/quantum-simulator/amazon/sv1")
    #     backend = AWSBraketBackend(
    #         device=device_wrapper.device,
    #         provider=AWSBraketProvider(),
    #         name=device_wrapper.device.name,
    #         description=f"AWS Device: {device_wrapper.device.provider_name} {device_wrapper.device.name}.",
    #         online_date=device_wrapper.device.properties.service.updatedAt,
    #         backend_version="2",
    #     )

    #     # Assert that the backend behaves as expected
    #     self.assertIsNotNone(backend)
    #     self.assertEqual(backend.name, device_wrapper.device.name)

    # @patch("modules.devices.braket.Ionq.Ionq")
    # @patch("qiskit_braket_provider.AWSBraketBackend")
    # def test_ionq_harmony(self, mock_aws_braket_backend, mock_ionq):
    #     mock_device_wrapper = MagicMock()
    #     mock_ionq.return_value = mock_device_wrapper

        # backend = self.library_instance.select_backend("ionQ_Harmony", 4)
        # mock_aws_braket_backend.assert_called_once()
        # self.assertEqual(backend, mock_aws_braket_backend.return_value)

    def test_invalid_configuration(self):
        with self.assertRaises(NotImplementedError) as context:
            self.library_instance.select_backend("invalid.backend", 4)
        self.assertIn("Device Configuration invalid.backend not implemented", str(context.exception))

    # These tests are commented out because:
    # - The complexity of mocking the behavior of Qiskit components (e.g., `transpile`, `Statevector`, and `AerSimulator`)
    #   makes it challenging to implement these tests in the current setup.
    # - The dependency on specific Qiskit modules and features requires more robust mocking strategies.
    # - We plan to revisit these tests in the future.
    # @patch("qiskit.transpiler.transpile")
    # @patch("qiskit.quantum_info.Statevector")
    # def test_aer_statevector_simulator(self, mock_statevector, mock_transpile):
    #     mock_circuit = MagicMock(spec=QuantumCircuit)
    #     mock_transpiled_circuit = MagicMock(spec=QuantumCircuit)
    #     mock_transpile.return_value = mock_transpiled_circuit
    #     mock_statevector.return_value.probabilities.return_value = np.array([0.25, 0.75])

    #     # Config
    #     config = "aer_statevector_simulator_gpu"
    #     config_dict = {"n_shots": 100}
    #     backend = MagicMock()

    #     execute_circuit, transpiled_circuit = self.library_instance.get_execute_circuit(
    #         mock_circuit, backend, config, config_dict
    #     )

    #     self.assertEqual(transpiled_circuit, mock_transpiled_circuit)
    #     solutions = [np.array([0.1, 0.9]), np.array([0.8, 0.2])]
    #     pmfs, samples = execute_circuit(solutions)

    #     # Validate the outputs
    #     self.assertIsInstance(pmfs, np.ndarray)
    #     self.assertIsNone(samples)
    #     self.assertEqual(pmfs.shape, (2, 2))
    #     np.testing.assert_array_equal(pmfs[0], [0.25, 0.75])

    # @patch("qiskit.transpile")
    # @patch("qiskit_aer.AerSimulator.run")
    # def test_aer_simulator(self, mock_run, mock_transpile):
    #     mock_circuit = MagicMock(spec=QuantumCircuit)
    #     mock_transpiled_circuit = MagicMock(spec=QuantumCircuit)
    #     mock_transpile.return_value = mock_transpiled_circuit
    #     mock_job = MagicMock()
    #     mock_job.result.return_value.get_counts.return_value.int_outcomes.return_value = {0: 10, 1: 20}
    #     mock_run.return_value = mock_job

    #     # Config
    #     mock_backend = MagicMock(spec=AerSimulator)
    #     mock_backend.version = 2
    #     config = "aer_simulator_gpu"
    #     config_dict = {"n_shots": 100}

    #     execute_circuit, transpiled_circuit = self.library_instance.get_execute_circuit(
    #         mock_circuit, mock_backend, config, config_dict
    #     )

    #     self.assertEqual(transpiled_circuit, mock_transpiled_circuit)
    #     solutions = [np.array([0.1, 0.9]), np.array([0.8, 0.2])]
    #     pmfs, samples = execute_circuit(solutions)

    #     self.assertIsInstance(pmfs, np.ndarray)
    #     self.assertIsInstance(samples, np.ndarray)
    #     self.assertEqual(pmfs.shape, (2, 2))
    #     self.assertEqual(samples.shape, (2, 2))
