import unittest

from modules.applications.optimization.scp.mappings.qubovertqubo import QubovertQUBO


class TestQubovertQUBO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.qubovert_instance = QubovertQUBO()
        cls.problem = (
            {1, 2, 3, 4},
            [{1, 2}, {2, 3}, {3, 4}]
        )
        cls.config = {"penalty_weight": 10.0}

    def test_get_requirements(self):
        requirements = self.qubovert_instance.get_requirements()
        expected_requirements = [{"name": "qubovert", "version": "1.2.5"}]
        for req in expected_requirements:
            self.assertIn(req, requirements)

    def test_get_parameter_options(self):
        options = self.qubovert_instance.get_parameter_options()
        self.assertIn("penalty_weight", options)
        self.assertIn("values", options["penalty_weight"])
        self.assertIn("description", options["penalty_weight"])

    def test_map(self):
        # Map the SCP problem to a QUBO representation
        qubo_mapping, mapping_time = self.qubovert_instance.map(self.problem, self.config)

        self.assertIn("Q", qubo_mapping)
        self.assertIsInstance(qubo_mapping["Q"], dict, "Expected 'Q' to be a dictionary.")
        self.assertGreater(len(qubo_mapping["Q"]), 0, "QUBO dictionary should not be empty.")
        self.assertGreater(mapping_time, 0, "Mapping time should be positive.")

    def test_reverse_map(self):
        # Map the SCP problem to a QUBO representation to populate self.SCP_problem
        self.qubovert_instance.map(self.problem, self.config)

        mock_solution = {0: 1, 1: 0, 2: 0}
        reverse_mapped_solution, reverse_mapping_time = self.qubovert_instance.reverse_map(mock_solution)

        self.assertIsInstance(reverse_mapped_solution, set)
        self.assertGreater(reverse_mapping_time, 0, "Reverse mapping time should be positive.")
        self.assertIn(0, reverse_mapped_solution, "Expected subset index 0 to be part of the solution.")

    def test_get_default_submodule(self):
        # Test valid submodule retrieval
        submodule = self.qubovert_instance.get_default_submodule("Annealer")
        self.assertIsNotNone(submodule, "Expected 'Annealer' submodule to be returned.")

        # Test invalid submodule option
        with self.assertRaises(NotImplementedError):
            self.qubovert_instance.get_default_submodule("InvalidSubmodule")
