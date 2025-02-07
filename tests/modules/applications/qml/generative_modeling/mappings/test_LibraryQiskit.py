import unittest
from unittest.mock import patch, MagicMock
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from modules.applications.qml.generative_modeling.mappings.LibraryQiskit import LibraryQiskit
from modules.applications.qml.generative_modeling.training.QCBM import QCBM
from modules.applications.qml.generative_modeling.training.QGAN import QGAN
from modules.applications.qml.generative_modeling.training.Inference import Inference


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

    def test_select_backend(self):
        with patch("qiskit_aer.Aer.get_backend", return_value=AerSimulator()) as mock_backend:
            backend = self.library_instance.select_backend("aer_simulator_cpu", 2)
            mock_backend.assert_called_once_with("aer_simulator")
            self.assertIsInstance(backend, AerSimulator)

        with self.assertRaises(NotImplementedError):
            self.library_instance.select_backend("unknown_backend", 2)

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

    def test_invalid_configuration(self):
        with self.assertRaises(NotImplementedError) as context:
            self.library_instance.select_backend("invalid.backend", 4)
        self.assertIn("Device Configuration invalid.backend not implemented", str(context.exception))

    def test_get_execute_circuit(self):
        circuit = QuantumCircuit(2)
        circuit.h(0)
        backend = AerSimulator()
        config_dict = {"n_shots": 100}

        execute_circuit, transpiled_circuit = self.library_instance.get_execute_circuit(
            circuit, backend, "aer_simulator_cpu", config_dict
        )

        self.assertIsNotNone(execute_circuit)
        self.assertIsInstance(transpiled_circuit, QuantumCircuit)
