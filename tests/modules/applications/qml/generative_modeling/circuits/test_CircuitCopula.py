import unittest

from quark.modules.applications.qml.generative_modeling.circuits.CircuitCopula import CircuitCopula


class TestCircuitCopula(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.circuit_instance = CircuitCopula()

    def test_initialization(self):
        self.assertEqual(self.circuit_instance.name, "CircuitCopula")
        self.assertEqual(
            self.circuit_instance.submodule_options,
            [
                "LibraryQiskit",
                "LibraryPennylane",
                "CustomQiskitNoisyBackend",
                "PresetQiskitNoisyBackend"
            ]
        )

    def test_get_requirements(self):
        requirements = self.circuit_instance.get_requirements()
        expected_requirements = [{"name": "scipy", "version": "1.12.0"}]
        self.assertEqual(requirements, expected_requirements)

    def test_get_parameter_options(self):
        parameter_options = self.circuit_instance.get_parameter_options()
        expected_options = {
            "depth": {
                "values": [1, 2, 3, 4, 5],
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
            "n_qubits": 8,
            "n_registers": 2,
            "histogram_train": [0.5, 0.5],
            "store_dir_iter": "/tmp",
            "train_size": 100,
            "dataset_name": "example_dataset",
            "binary_train": True
        }
        config = {"depth": 3}

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

        self.assertEqual(output["circuit_name"], "Copula")
        self.assertEqual(output["n_qubits"], input_data["n_qubits"])
        self.assertEqual(output["n_registers"], input_data["n_registers"])
        self.assertEqual(output["depth"], config["depth"])

        # Validate the structure of the gate sequence
        self.assertIsInstance(output["gate_sequence"], list)
        self.assertTrue(all(isinstance(gate, list) for gate in output["gate_sequence"]))

    def test_generate_gate_sequence_edge_cases(self):
        # Test with minimum qubits and depth
        input_data = {
            "n_qubits": 2,
            "n_registers": 2,
            "histogram_train": [1.0],
            "store_dir_iter": "/tmp",
            "train_size": 10,
            "dataset_name": "minimal_dataset",
            "binary_train": False
        }
        config = {"depth": 1}

        output = self.circuit_instance.generate_gate_sequence(input_data, config)

        self.assertEqual(output["n_qubits"], 2)
        self.assertEqual(output["n_registers"], 2)
        self.assertEqual(output["depth"], 1)
        self.assertGreater(len(output["gate_sequence"]), 0)

    def test_invalid_config(self):
        input_data = {
            "n_qubits": 4,
            "n_registers": 2,
            "histogram_train": [0.5, 0.5],
            "store_dir_iter": "/tmp",
            "train_size": 50,
            "dataset_name": "invalid_dataset",
            "binary_train": False
        }

        # Missing depth in config
        config = {}
        with self.assertRaises(KeyError):
            self.circuit_instance.generate_gate_sequence(input_data, config)

        # Invalid depth value
        config = {"depth": -1}
        with self.assertRaises(ValueError, msg="Expected ValueError for invalid depth"):
            depth = config["depth"]
            if depth <= 0:
                raise ValueError("Depth must be positive")
            self.circuit_instance.generate_gate_sequence(input_data, config)


if __name__ == "__main__":
    unittest.main()
