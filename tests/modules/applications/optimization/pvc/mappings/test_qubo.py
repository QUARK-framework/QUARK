import unittest
import pickle

from src.modules.applications.optimization.pvc.mappings.qubo import QUBO


class TestQUBO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.qubo_instance = QUBO()
        with open("tests/modules/applications/optimization/pvc/mappings/pvc_graph_1_seam.gpickle", "rb") as file:
            cls.graph = pickle.load(file)

    def test_get_requirements(self):
        requirements = self.qubo_instance.get_requirements()
        expected_requirements = [{"name": "networkx", "version": "3.4.2"}]
        self.assertEqual(requirements, expected_requirements)

    def test_get_parameter_options(self):
        parameter_options = self.qubo_instance.get_parameter_options()
        expected_options = {
            "lagrange_factor": {
                "values": [0.75, 1.0, 1.25],
                "description": "By which factor would you like to multiply your Lagrange?"
            }
        }
        self.assertEqual(parameter_options, expected_options)

    def test_map(self):
        config = {"lagrange_factor": 1.0}

        qubo_mapping, mapping_time = self.qubo_instance.map(self.graph, config)

        # Check Q dictionary presence and type
        self.assertIn("Q", qubo_mapping)
        self.assertIsInstance(qubo_mapping["Q"], dict)

        # Ensure QUBO matrix contains entries
        q_matrix = qubo_mapping["Q"]
        self.assertGreater(len(q_matrix), 0, "QUBO matrix should contain entries")

        # Confirm tuple keys and float values in QUBO matrix
        for key, value in q_matrix.items():
            self.assertIsInstance(key, tuple)
            self.assertIsInstance(value, float)

        # Confirm timing
        self.assertGreater(mapping_time, 0, "Mapping time should be greater than zero")

    def test_get_default_submodule(self):
        submodule = self.qubo_instance.get_default_submodule("Annealer")
        self.assertIsNotNone(submodule, "Annealer submodule should not be None")

        with self.assertRaises(NotImplementedError):
            self.qubo_instance.get_default_submodule("InvalidSubmodule")
