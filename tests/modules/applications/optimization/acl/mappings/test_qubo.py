import unittest
import numpy as np
from qiskit_optimization import QuadraticProgram

from modules.applications.optimization.acl.mappings.qubo import Qubo


class TestQubo(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.qubo_instance = Qubo()
        # Mock problem dictionary for testing
        cls.problem_dict = {
            "variables": [
                {"name": "x_0", "cat": "Integer", "lowBound": 0, "upBound": 1},
                {"name": "x_1", "cat": "Integer", "lowBound": 0, "upBound": 1},
            ],
            "objective": {
                "coefficients": [
                    {"name": "x_0", "value": 3},
                    {"name": "x_1", "value": 4},
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
        requirements = self.qubo_instance.get_requirements()
        expected_requirements = [
            {"name": "numpy", "version": "1.26.4"},
            {"name": "qiskit-optimization", "version": "0.6.1"},
        ]
        for req in expected_requirements:
            self.assertIn(req, requirements)

    def test_get_parameter_options(self):
        options = self.qubo_instance.get_parameter_options()
        self.assertEqual(options, {}, "Expected parameter options to be an empty dictionary.")

    def test_map_pulp_to_qiskit(self):
        qp = self.qubo_instance.map_pulp_to_qiskit(self.problem_dict)
        self.assertIsInstance(qp, QuadraticProgram, "Expected a QuadraticProgram instance.")
        self.assertEqual(len(qp.variables), 2, "Incorrect number of variables in QuadraticProgram.")
        self.assertEqual(len(qp.linear_constraints), 1, "Incorrect number of constraints in QuadraticProgram.")

    def test_convert_string_to_arguments(self):
        qubo_string = "maximize 3*x_0 + 4*x_1 - 2*x_0*x_1"
        arguments = self.qubo_instance.convert_string_to_arguments(qubo_string)
        expected_arguments = [
            [3.0, "x_0"],
            [4.0, "x_1"],
            [-2.0, "x_0", "x_1"]
        ]
        self.assertEqual(arguments, expected_arguments, "String to arguments conversion failed.")

    def test_construct_qubo(self):
        arguments = [
            [3.0, "x_0"],
            [4.0, "x_1"],
            [-2.0, "x_0", "x_1"]
        ]
        variables = ["x_0", "x_1"]
        qubo_matrix = self.qubo_instance.construct_qubo(arguments, variables)
        expected_matrix = np.array([
            [-3.0, 0.0],  # Diagonal terms and no interaction (negative due to minimization problem)
            [2.0, -4.0]   # Off-diagonal term and diagonal term for x_1
        ])
        np.testing.assert_array_almost_equal(qubo_matrix, expected_matrix, err_msg="QUBO matrix construction failed.")

    def test_map(self):
        qubo_mapping, mapping_time = self.qubo_instance.map(self.problem_dict, self.config)
        self.assertIn("Q", qubo_mapping, "Expected 'Q' in QUBO mapping.")
        self.assertIsInstance(qubo_mapping["Q"], np.ndarray, "Expected 'Q' to be a numpy array.")
        self.assertGreater(mapping_time, 0, "Mapping time should be positive.")

    def test_reverse_map(self):
        mock_solution = {0: 1, 1: 0}  # Example binary solution
        reverse_mapped_solution, reverse_mapping_time = self.qubo_instance.reverse_map(mock_solution)
        self.assertIsInstance(reverse_mapped_solution, dict, "Expected a dictionary as the reverse mapping result.")
        self.assertIn("variables", reverse_mapped_solution, "Expected 'variables' in reverse mapping result.")
        self.assertGreater(reverse_mapping_time, 0, "Reverse mapping time should be positive.")

    def test_get_default_submodule(self):
        submodule = self.qubo_instance.get_default_submodule("Annealer")
        self.assertIsNotNone(submodule, "Expected 'Annealer' submodule to be returned.")
        with self.assertRaises(NotImplementedError):
            self.qubo_instance.get_default_submodule("InvalidSubmodule")
