import unittest
import numpy as np
from qiskit_optimization import QuadraticProgram

from src.modules.applications.optimization.acl.mappings.ising import Ising


class TestIsing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.ising_instance = Ising()
        # Mock problem dictionary for testing
        cls.problem_dict = {
            "variables": [
                {"name": "x_0", "cat": "Integer", "lowBound": 0, "upBound": 1},
                {"name": "x_1", "cat": "Integer", "lowBound": 0, "upBound": 10},
            ],
            "objective": {
                "coefficients": [
                    {"name": "x_0", "value": 1},
                    {"name": "x_1", "value": 2},
                ]
            },
            "constraints": [
                {
                    "name": "c1",
                    "coefficients": [
                        {"name": "x_0", "value": 1},
                        {"name": "x_1", "value": 1},
                    ],
                    "sense": -1,
                    "constant": -1,
                }
            ],
            "parameters": {"sense": -1},  # Maximize
        }
        cls.config = {}

    def test_get_requirements(self):
        requirements = self.ising_instance.get_requirements()
        expected_requirements = [
            {"name": "numpy", "version": "1.26.4"},
            {"name": "more-itertools", "version": "10.5.0"},
            {"name": "qiskit-optimization", "version": "0.6.1"},
        ]
        for req in expected_requirements:
            self.assertIn(req, requirements)

    def test_get_parameter_options(self):
        options = self.ising_instance.get_parameter_options()
        self.assertEqual(options, {}, "Expected parameter options to be an empty dictionary.")

    def test_map_pulp_to_qiskit(self):
        qp = self.ising_instance.map_pulp_to_qiskit(self.problem_dict)
        self.assertIsInstance(qp, QuadraticProgram, "Expected a QuadraticProgram instance.")
        self.assertEqual(len(qp.variables), 2, "Incorrect number of variables in QuadraticProgram.")
        self.assertEqual(len(qp.linear_constraints), 1, "Incorrect number of constraints in QuadraticProgram.")

    def test_map(self):
        ising_mapping, mapping_time = self.ising_instance.map(self.problem_dict, self.config)
        self.assertIn("J", ising_mapping, "Expected 'J' in Ising mapping.")
        self.assertIn("t", ising_mapping, "Expected 't' in Ising mapping.")
        self.assertGreater(mapping_time, 0, "Mapping time should be positive.")

    def test_reverse_map(self):
        mock_solution = {0: 1, 1: 0}  # Example binary solution

        reverse_mapped_solution, reverse_mapping_time = self.ising_instance.reverse_map(mock_solution)
        self.assertIsInstance(reverse_mapped_solution, dict, "Expected a dictionary as the reverse mapping result.")
        self.assertIn("variables", reverse_mapped_solution, "Expected 'variables' in reverse mapping result.")
        self.assertGreater(reverse_mapping_time, 0, "Reverse mapping time should be positive.")

    def test_convert_ising_to_qubo(self):
        solution = np.array([-1, 1, -1, 1])
        expected_result = np.array([0, 1, 0, 1])
        converted_solution = self.ising_instance._convert_ising_to_qubo(solution)
        self.assertTrue(np.array_equal(converted_solution, expected_result), "Conversion to QUBO failed.")

    def test_get_default_submodule(self):
        submodule = self.ising_instance.get_default_submodule("QAOA")
        self.assertIsNotNone(submodule, "Expected 'QAOA' submodule to be returned.")
        with self.assertRaises(NotImplementedError):
            self.ising_instance.get_default_submodule("InvalidSubmodule")
