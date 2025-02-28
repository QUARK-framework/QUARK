import unittest

from nnf import And, Or, Var
from pysat.formula import WCNF

from modules.applications.optimization.sat.mappings.direct import Direct


class TestDirect(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.direct_instance = Direct()

        # Hard constraints as CNF: Conjunction of disjunctions
        cls.hard_constraints = And([Or([Var("L0")]), Or([~Var("L1")])])  # A ∧ ¬B

        # Soft constraints as CNF: List of disjunctions
        cls.soft_constraints = [Or([Var("L2")]), Or([~Var("L3")])]
        cls.problem = (cls.hard_constraints, cls.soft_constraints)

    def test_get_requirements(self):
        requirements = self.direct_instance.get_requirements()
        expected_requirements = [
            {"name": "nnf", "version": "0.4.1"},
            {"name": "python-sat", "version": "1.8.dev13"},
        ]
        for req in expected_requirements:
            self.assertIn(req, requirements)

    def test_get_parameter_options(self):
        options = self.direct_instance.get_parameter_options()
        self.assertEqual(options, {}, "Expected parameter options to be an empty dictionary.")

    def test_map(self):
        # Map the SAT problem to pysat
        mapped_problem, mapping_time = self.direct_instance.map(self.problem, config={})

        # Check that the result is a WCNF instance
        self.assertIsInstance(mapped_problem, WCNF, "Expected a WCNF instance.")
        self.assertGreater(mapping_time, 0, "Mapping time should be positive.")

        # Check the number of hard and soft constraints
        self.assertEqual(len(mapped_problem.hard), 2, "Expected 1 hard constraint.")
        self.assertEqual(len(mapped_problem.soft), 2, "Expected 2 soft constraints.")

    def test_reverse_map(self):
        solution = [1, -2, 3, -4, 5]  # Example literals
        reverse_mapped_solution, reverse_mapping_time = self.direct_instance.reverse_map(solution)

        # Check that the result is a dictionary
        self.assertIsInstance(
            reverse_mapped_solution, dict, "Expected a dictionary as the reverse mapping result."
        )
        self.assertGreater(reverse_mapping_time, 0, "Reverse mapping time should be positive.")

        # Verify that the literals are correctly mapped
        expected_solution = {
            "L0": True,  # L1 (positive)
            "L1": False,  # L2 (negative)
            "L2": True,  # L3 (positive)
            "L3": False,  # L4 (negative)
            "L4": True,  # L5 (positive)
        }
        self.assertEqual(reverse_mapped_solution, expected_solution, "Reverse mapping result is incorrect.")

    def test_get_default_submodule(self):
        # Test valid submodules
        submodule = self.direct_instance.get_default_submodule("ClassicalSAT")
        self.assertIsNotNone(submodule, "Expected 'ClassicalSAT' submodule to be returned.")

        submodule = self.direct_instance.get_default_submodule("RandomSAT")
        self.assertIsNotNone(submodule, "Expected 'RandomSAT' submodule to be returned.")

        # Test invalid submodule option
        with self.assertRaises(NotImplementedError):
            self.direct_instance.get_default_submodule("InvalidSubmodule")
