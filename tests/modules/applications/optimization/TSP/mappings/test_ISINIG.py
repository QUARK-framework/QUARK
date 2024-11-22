import unittest
import networkx as nx
import numpy as np

from modules.applications.optimization.TSP.mappings.ISING import Ising


class TestIsing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ising_instance = Ising()
        # Sample TSP graph with 4 nodes
        cls.graph = nx.complete_graph(4)
        for (u, v) in cls.graph.edges():
            cls.graph[u][v]['weight'] = np.random.randint(1, 10)
        cls.config = {"lagrange_factor": 1.0, "mapping": "pyqubo"}

    def test_get_requirements(self):
        requirements = self.ising_instance.get_requirements()
        expected_requirements = [
            {"name": "networkx", "version": "3.2.1"},
            {"name": "numpy", "version": "1.26.4"},
            {"name": "dimod", "version": "0.12.17"},
            {"name": "more-itertools", "version": "10.5.0"},
            {"name": "qiskit-optimization", "version": "0.6.1"},
            {"name": "pyqubo", "version": "1.4.0"},
        ]
        for req in expected_requirements:
            self.assertIn(req, requirements)

    def test_get_parameter_options(self):
        options = self.ising_instance.get_parameter_options()
        self.assertIn("lagrange_factor", options)
        self.assertIn("mapping", options)

    def test_map_pyqubo(self):
        self.config["mapping"] = "pyqubo"
        ising_mapping, mapping_time = self.ising_instance.map(self.graph, self.config)
        self.assertIn("J", ising_mapping)
        self.assertIn("J_dict", ising_mapping)
        self.assertIn("t", ising_mapping)

        self.assertIsInstance(ising_mapping["J"], np.ndarray)
        self.assertIsInstance(ising_mapping["t"], np.ndarray)
        self.assertGreater(mapping_time, 0, "Mapping time should be positive.")

    def test_map_ocean(self):
        self.config["mapping"] = "ocean"
        ising_mapping, mapping_time = self.ising_instance.map(self.graph, self.config)
        self.assertIn("J", ising_mapping)
        self.assertIn("J_dict", ising_mapping)
        self.assertIn("t", ising_mapping)

        self.assertIsInstance(ising_mapping["J"], np.ndarray)
        self.assertIsInstance(ising_mapping["t"], np.ndarray)
        self.assertGreater(mapping_time, 0, "Mapping time should be positive.")

    def test_map_qiskit(self):
        self.config["mapping"] = "qiskit"
        ising_mapping, mapping_time = self.ising_instance.map(self.graph, self.config)
        self.assertIn("J", ising_mapping)
        self.assertIn("t", ising_mapping)

        self.assertIsInstance(ising_mapping["J"], np.ndarray)
        self.assertIsInstance(ising_mapping["t"], np.ndarray)
        self.assertGreater(mapping_time, 0, "Mapping time should be positive.")

    def test_reverse_map(self):
        self.config["mapping"] = "ocean"
        self.ising_instance.map(self.graph, self.config)

        # Create a mock solution with binary values
        solution_size = len(self.ising_instance.key_mapping)
        mock_solution = [1 if i % 2 == 0 else 0 for i in range(solution_size)]
        reverse_mapped_solution, reverse_mapping_time = self.ising_instance.reverse_map(mock_solution)
        
        self.assertIsInstance(reverse_mapped_solution, dict)
        self.assertGreater(reverse_mapping_time, 0, "Reverse mapping time should be positive.")
    
    def test_flip_bits_in_bitstring(self):
        # Input with alternating 0s and 1s
        solution = [0, 1, 0, 1, 0, 1]
        expected_flipped = [1, 0, 1, 0, 1, 0]
        flipped_solution = self.ising_instance._flip_bits_in_bitstring(solution)
        self.assertTrue(np.array_equal(flipped_solution, expected_flipped), "Bits were not flipped correctly.")

        # Input with all 0s
        solution = [0, 0, 0, 0]
        expected_flipped = [1, 1, 1, 1]
        flipped_solution = self.ising_instance._flip_bits_in_bitstring(solution)
        self.assertTrue(np.array_equal(flipped_solution, expected_flipped), "All 0s were not flipped to 1s.")

        # Input with all 1s
        solution = [1, 1, 1, 1]
        expected_flipped = [0, 0, 0, 0]
        flipped_solution = self.ising_instance._flip_bits_in_bitstring(solution)
        self.assertTrue(np.array_equal(flipped_solution, expected_flipped), "All 1s were not flipped to 0s.")

    def test_convert_ising_to_qubo(self):
        # Input with -1s and 1s
        solution = [-1, 1, -1, 1]
        expected_converted = [0, 1, 0, 1]
        qubo_solution = self.ising_instance._convert_ising_to_qubo(solution)
        self.assertTrue(np.array_equal(qubo_solution, expected_converted), "Ising to QUBO conversion failed for -1s and 1s.")

        # Input with no -1s
        solution = [0, 1, 0, 1]
        expected_converted = [0, 1, 0, 1]
        qubo_solution = self.ising_instance._convert_ising_to_qubo(solution)
        self.assertTrue(np.array_equal(qubo_solution, expected_converted), "QUBO solution should remain unchanged.")

        # Input with only -1s
        solution = [-1, -1, -1, -1]
        expected_converted = [0, 0, 0, 0]
        qubo_solution = self.ising_instance._convert_ising_to_qubo(solution)
        self.assertTrue(np.array_equal(qubo_solution, expected_converted), "Ising to QUBO conversion failed for all -1s.")


    def test_get_default_submodule(self):
        # Test valid submodules
        submodules = ["QAOA", "PennylaneQAOA", "QiskitQAOA"]
        for option in submodules:
            with self.subTest(option=option):
                submodule = self.ising_instance.get_default_submodule(option)
                self.assertIsNotNone(submodule, f"{option} submodule should not be None")

        # Test invalid submodule option
        with self.assertRaises(NotImplementedError):
            self.ising_instance.get_default_submodule("InvalidSubmodule")
