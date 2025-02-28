import pickle
import unittest
<<<<<<< HEAD:tests/modules/applications/optimization/pvc/mappings/test_ising.py
=======

import numpy as np
>>>>>>> GreshmaShaji-binpacking_and_mipsolver:tests/modules/applications/optimization/PVC/mappings/test_ISING.py

import numpy as np

from modules.applications.optimization.pvc.mappings.ising import Ising
from modules.applications.optimization.pvc.mappings.qubo import QUBO


class TestIsing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ising_instance = Ising()
        with open("tests/modules/applications/optimization/pvc/mappings/pvc_graph_1_seam.gpickle", "rb") as file:
            cls.graph = pickle.load(file)

    def test_get_requirements(self):
        requirements = self.ising_instance.get_requirements()
        expected_requirements = [
            {"name": "networkx", "version": "3.4.2"},
            {"name": "numpy", "version": "1.26.4"},
            {"name": "dimod", "version": "0.12.18"},
            *QUBO.get_requirements()
        ]

        self.assertEqual(requirements, expected_requirements)

    def test_get_parameter_options(self):
        parameter_options = self.ising_instance.get_parameter_options()
        expected_options = {
            "lagrange_factor": {
                "values": [0.75, 1.0, 1.25],
                "description": "By which factor would you like to multiply your Lagrange?"
            }
        }
        self.assertEqual(parameter_options, expected_options)

    def test_map(self):
        config = {"lagrange_factor": 1.0}
        # Run map function
        ising_mapping, mapping_time = self.ising_instance.map(self.graph, config)

        self.assertIn("J", ising_mapping)
        self.assertIn("t", ising_mapping)
        self.assertIsInstance(ising_mapping["J"], np.ndarray)
        self.assertIsInstance(ising_mapping["t"], np.ndarray)
        self.assertGreater(mapping_time, 0, "Mapping time should be greater than zero.")

        j_matrix_shape = ising_mapping["J"].shape
        self.assertEqual(j_matrix_shape[0], j_matrix_shape[1], "J matrix should be square.")
        self.assertGreater(j_matrix_shape[0], 0, "J matrix should have positive dimensions.")
        self.assertEqual(len(ising_mapping["t"]), j_matrix_shape[0], "t vector length should match J matrix size.")

    def test_reverse_map(self):
        self.ising_instance.key_mapping = {(0, 0): 0, (1, 1): 1}
        mock_solution = {0: 1, 1: 0}

        reverse_mapped_solution, reverse_mapping_time = self.ising_instance.reverse_map(mock_solution)

        expected_solution = {(0, 0): 1, (1, 1): 0}
        self.assertEqual(reverse_mapped_solution, expected_solution)
        self.assertGreater(reverse_mapping_time, 0, "Reverse mapping time should be greater than zero.")

    def test_get_default_submodule(self):
        submodule = self.ising_instance.get_default_submodule("QAOA")
        self.assertIsNotNone(submodule, "QAOA submodule should not be None")

        submodule = self.ising_instance.get_default_submodule("PennylaneQAOA")
        self.assertIsNotNone(submodule, "PennylaneQAOA submodule should not be None")
