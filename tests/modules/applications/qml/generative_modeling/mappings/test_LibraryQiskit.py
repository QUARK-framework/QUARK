import unittest
from unittest.mock import patch, MagicMock
from qiskit import QuantumCircuit, transpile
from qiskit.providers import Backend
from qiskit.quantum_info import Statevector
import numpy as np

from qiskit.transpiler.target import Target
from qiskit.transpiler import InstructionProperties
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
            {"name": "qiskit", "version": "1.1.0"},
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

    # @patch("qiskit_aer.Aer.get_backend")
    # @patch("modules.devices.braket.Ionq")
    # @patch("qiskit_braket_provider.AWSBraketBackend")
    # def test_select_backend(self, mock_braket_backend, mock_ionq, mock_get_backend):
    #     # Mocking Qiskit Aer backend
    #     mock_backend = MagicMock()
    #     mock_get_backend.return_value = mock_backend

    #     # Test aer_simulator_cpu
    #     backend = self.library_instance.select_backend("aer_simulator_cpu", 2)
    #     mock_get_backend.assert_called_with("aer_simulator")
    #     mock_backend.set_options.assert_called_with(device="CPU")
    #     self.assertIs(mock_backend, backend)

    #     # Test aer_simulator_gpu
    #     backend = self.library_instance.select_backend("aer_simulator_gpu", 2)
    #     mock_get_backend.assert_called_with("aer_simulator")
    #     mock_backend.set_options.assert_called_with(device="GPU")
    #     self.assertIs(mock_backend, backend)

    #     # Test aer_statevector_simulator_cpu
    #     backend = self.library_instance.select_backend("aer_statevector_simulator_cpu", 3)
    #     mock_get_backend.assert_called_with("statevector_simulator")
    #     mock_backend.set_options.assert_called_with(device="CPU")
    #     self.assertIs(mock_backend, backend)

    #     # Test aer_statevector_simulator_gpu
    #     backend = self.library_instance.select_backend("aer_statevector_simulator_gpu", 3)
    #     mock_get_backend.assert_called_with("statevector_simulator")
    #     mock_backend.set_options.assert_called_with(device="GPU")
    #     self.assertIs(mock_backend, backend)

    #     # Mocking IonQ backend
    #     mock_device = MagicMock()
    #     mock_ionq.return_value.device = mock_device
    #     backend = self.library_instance.select_backend("ionQ_Harmony", 3)
    #     self.assertIsInstance(backend, MagicMock)
    #     mock_ionq.assert_called_with("ionQ", "arn:aws:braket:::device/qpu/ionq/ionQdevice")

    #     # Test Amazon_SV1 backend
    #     backend = self.library_instance.select_backend("Amazon_SV1", 4)
    #     self.assertIsInstance(backend, MagicMock)

    #     # Test invalid backend
    #     with self.assertRaises(NotImplementedError):
    #         self.library_instance.select_backend("invalid_backend", 2)


    # @patch("qiskit.quantum_info.Statevector.probabilities")
    # @patch("qiskit.transpile")
    # def test_get_execute_circuit(self, mock_transpile, mock_probabilities):
    #     # Mock transpile function
    #     mock_transpiled_circuit = QuantumCircuit(2)
    #     mock_transpile.return_value = mock_transpiled_circuit

    #     # Mock probabilities function
    #     mock_probabilities.return_value = [0.5, 0.5]

    #     # Create a mock backend
    #     mock_backend = MagicMock()
    #     mock_backend.version = 1
    #     mock_backend.num_qubits = 2

    #     # Mock qubit properties to match num_qubits
    #     mock_backend.qubit_properties = [None] * mock_backend.num_qubits

    #     # Create a mock Target with mocked instructions
    #     mock_target = Target(num_qubits=2)
    #     mock_target.add_instruction("rx", InstructionProperties())
    #     mock_target.add_instruction("ry", InstructionProperties())
    #     mock_target.add_instruction("rz", InstructionProperties())
    #     mock_target.add_instruction("cx", InstructionProperties())
    #     mock_target.add_instruction("measure", InstructionProperties())
    #     mock_backend.target = mock_target

    #     # Add configuration mock
    #     mock_backend.configuration = MagicMock()
    #     mock_backend.configuration.num_qubits = mock_backend.num_qubits

    #     # Test inputs
    #     config_dict = {"n_shots": 100}
    #     circuit = QuantumCircuit(2)

    #     # Call get_execute_circuit
    #     execute_circuit, transpiled_circuit = self.library_instance.get_execute_circuit(
    #         circuit, mock_backend, "aer_statevector_simulator_cpu", config_dict
    #     )

    #     # Assertions
    #     self.assertIs(transpiled_circuit, mock_transpiled_circuit)
    #     self.assertTrue(callable(execute_circuit))

    #     # Test execution with mock data
    #     solutions = [np.array([1.0, 2.0])]
    #     pmfs, _ = execute_circuit(solutions)

    #     self.assertIsInstance(pmfs, np.ndarray)
    #     self.assertEqual(len(pmfs), len(solutions))
    #     self.assertAlmostEqual(np.sum(pmfs[0]), 1.0, places=6)





if __name__ == "__main__":
    unittest.main()
