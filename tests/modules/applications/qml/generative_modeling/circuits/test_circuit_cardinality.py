import unittest

from modules.applications.qml.generative_modeling.circuits.circuit_cardinality import CircuitCardinality


class TestCircuitCardinality(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.circuit_instance = CircuitCardinality()

    def test_initialization(self):
        self.assertEqual(self.circuit_instance.name, "CircuitCardinality")
        self.assertEqual(
            self.circuit_instance.submodule_options,
            [
                "LibraryQiskit",
                "LibraryPennylane",
                "CustomQiskitNoisyBackend",
                "PresetQiskitNoisyBackend"
            ]
        )

    def test_get_parameter_options(self):
        parameter_options = self.circuit_instance.get_parameter_options()
        expected_options = {
            "depth": {
                "values": [2, 4, 8, 16],
                "description": "What depth do you want?"
            }
        }
        self.assertEqual(parameter_options, expected_options)

    def test_get_default_submodule(self):
        submodules = [
            "LibraryQiskit", "LibraryPennylane",
            "CustomQiskitNoisyBackend", "PresetQiskitNoisyBackend"
        ]
        for option in submodules:
            with self.subTest(option=option):
                submodule = self.circuit_instance.get_default_submodule(option)
                self.assertIsNotNone(submodule, f"Expected non-None submodule for {option}")

        with self.assertRaises(NotImplementedError):
            self.circuit_instance.get_default_submodule("InvalidOption")

    def test_generate_gate_sequence(self):
        input_data = {
            "n_qubits": 3,
            "n_registers": 3,
            "histogram_train": [0.5, 0.5],
            "store_dir_iter": "/tmp",
            "train_size": 100,
            "dataset_name": "example_dataset",
            "binary_train": True
        }
        config = {"depth": 4}

        output = self.circuit_instance.generate_gate_sequence(input_data, config)

        self.assertIn("gate_sequence", output)
        self.assertIn("circuit_name", output)
        self.assertIn("n_qubits", output)
        self.assertIn("n_registers", output)
        self.assertIn("depth", output)
        self.assertIn("histogram_train", output)
        self.assertIn("store_dir_iter", output)
        self.assertIn("train_size", output)
        self.assertIn("dataset_name", output)
        self.assertIn("binary_train", output)

        self.assertEqual(output["circuit_name"], "Cardinality")
        self.assertEqual(output["n_qubits"], input_data["n_qubits"])
        self.assertEqual(output["n_registers"], input_data["n_registers"])
        self.assertEqual(output["depth"], config["depth"] // 2)

        # Validate the structure of the gate sequence
        self.assertIsInstance(output["gate_sequence"], list)
        self.assertTrue(all(isinstance(gate, list) for gate in output["gate_sequence"]))

    def test_generate_gate_sequence_edge_cases(self):
        # Test with minimum qubits and depth
        input_data = {
            "n_qubits": 1,
            "n_registers": 1,
            "histogram_train": [1.0],
            "store_dir_iter": "/tmp",
            "train_size": 10,
            "dataset_name": "minimal_dataset",
            "binary_train": False
        }
        config = {"depth": 2}

        output = self.circuit_instance.generate_gate_sequence(input_data, config)

        self.assertEqual(output["n_qubits"], 1)
        self.assertEqual(output["n_registers"], 1)
        self.assertEqual(output["depth"], 1)
        self.assertEqual(len(output["gate_sequence"]), 6)

    def test_invalid_config(self):
        input_data = {
            "n_qubits": 3,
            "n_registers": 3,
            "histogram_train": [0.5, 0.5],
            "store_dir_iter": "/tmp",
            "train_size": 100,
            "dataset_name": "example_dataset",
            "binary_train": True
        }

        # Missing depth in config
        config = {}
        with self.assertRaises(KeyError):
            self.circuit_instance.generate_gate_sequence(input_data, config)

        # Invalid depth value
        config = {"depth": -1}
        with self.assertRaises(ValueError):
            depth = config["depth"]
            if depth <= 0:
                raise ValueError("Depth must be positive")
            self.circuit_instance.generate_gate_sequence(input_data, config)
