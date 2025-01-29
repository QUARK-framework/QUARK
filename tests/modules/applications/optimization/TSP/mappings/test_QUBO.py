import unittest
import networkx as nx

from quark.modules.applications.optimization.TSP.mappings.QUBO import QUBO


class TestQUBO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.qubo_instance = QUBO()
        # Define a sample TSP graph with 4 nodes
        cls.graph = nx.complete_graph(4)
        for (u, v) in cls.graph.edges():
            cls.graph[u][v]['weight'] = 1
        cls.config = {"lagrange_factor": 1.0}

    def test_get_requirements(self):
        requirements = self.qubo_instance.get_requirements()
        expected_requirements = [
            {"name": "networkx", "version": "3.4.2"},
            {"name": "dwave_networkx", "version": "0.8.15"},
        ]
        for req in expected_requirements:
            self.assertIn(req, requirements)

    def test_get_parameter_options(self):
        options = self.qubo_instance.get_parameter_options()
        self.assertIn("lagrange_factor", options)
        self.assertTrue("values" in options["lagrange_factor"])
        self.assertTrue("postproc" in options["lagrange_factor"])

    def test_map(self):
        # Test mapping a TSP problem to QUBO
        qubo_mapping, mapping_time = self.qubo_instance.map(self.graph, self.config)
        self.assertIn("Q", qubo_mapping, "Expected 'Q' key in QUBO mapping.")
        self.assertIsInstance(qubo_mapping["Q"], dict, "Expected 'Q' to be a dictionary.")
        self.assertGreater(mapping_time, 0, "Mapping time should be positive.")

    def test_get_default_submodule(self):
        # Test valid submodule retrieval
        submodule = self.qubo_instance.get_default_submodule("Annealer")
        self.assertIsNotNone(submodule, "Expected 'Annealer' submodule to be returned.")

        # Test invalid submodule option
        with self.assertRaises(NotImplementedError):
            self.qubo_instance.get_default_submodule("InvalidSubmodule")
