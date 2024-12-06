import unittest
from nnf import Var, And, Or

from modules.applications.optimization.SAT.mappings.ChoiQUBO import ChoiQUBO


class TestChoiQUBO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.choi_qubo_instance = ChoiQUBO()
        # Define a problem with both hard and soft constraints
        hard_constraints = And([Or([Var("L1"), ~Var("L2")]), Or([~Var("L3"), Var("L4")])])
        soft_constraints = [Or([Var("L5"), ~Var("L6")])]
        cls.problem = (hard_constraints, soft_constraints)
        cls.config = {"hard_reward": 0.9, "soft_reward": 1.0}

    def test_get_requirements(self):
        requirements = self.choi_qubo_instance.get_requirements()
        self.assertIn({"name": "nnf", "version": "0.4.1"}, requirements)

    def test_get_parameter_options(self):
        options = self.choi_qubo_instance.get_parameter_options()
        self.assertIn("hard_reward", options)
        self.assertIn("soft_reward", options)

    def test_map(self):
        qubo_mapping, mapping_time = self.choi_qubo_instance.map(self.problem, self.config)

        self.assertIn("Q", qubo_mapping)

        q_matrix = qubo_mapping["Q"]
        self.assertIsInstance(q_matrix, dict)
        for key, value in q_matrix.items():
            self.assertIsInstance(key, tuple, "Expected key to be a tuple")
            self.assertTrue(isinstance(
                value, float) or isinstance(value, int), "Expected value to be a float or int"
            )

        self.assertGreater(mapping_time, 0, "Mapping time should be greater than zero.")

    def test_reverse_map(self):
        mock_solution = {i: 1 if i % 2 == 0 else 0 for i in range(len(self.choi_qubo_instance.reverse_dict))}

        reverse_mapped_solution, reverse_mapping_time = self.choi_qubo_instance.reverse_map(mock_solution)

        self.assertIsInstance(reverse_mapped_solution, dict)
        self.assertGreater(reverse_mapping_time, 0, "Reverse mapping time should be greater than zero.")

    def test_get_default_submodule(self):
        submodule = self.choi_qubo_instance.get_default_submodule("Annealer")
        self.assertIsNotNone(submodule, "Annealer submodule should not be None")

        with self.assertRaises(NotImplementedError):
            self.choi_qubo_instance.get_default_submodule("InvalidSubmodule")
