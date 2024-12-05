import unittest
import numpy as np
from nnf import Var, And, Or
from modules.applications.optimization.SAT.mappings.ChoiISING import ChoiIsing


class TestChoiIsing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.choi_ising_instance = ChoiIsing()
        hard_constraints = And([Or([Var("L1"), ~Var("L2")]), Or([~Var("L3"), Var("L4")])])
        soft_constraints = [Or([Var("L5"), ~Var("L6")])]
        cls.problem = (hard_constraints, soft_constraints)
        cls.config = {"hard_reward": 0.9, "soft_reward": 1.0}

    def test_get_requirements(self):
        requirements = self.choi_ising_instance.get_requirements()
        self.assertIn({"name": "numpy", "version": "1.26.4"}, requirements)
        self.assertIn({"name": "dimod", "version": "0.12.18"}, requirements)

    def test_get_parameter_options(self):
        options = self.choi_ising_instance.get_parameter_options()
        self.assertIn("hard_reward", options)
        self.assertIn("soft_reward", options)

    def test_map(self):
        ising_mapping, mapping_time = self.choi_ising_instance.map(self.problem, self.config)

        # Check that mapping results contain the expected keys
        self.assertIn("J", ising_mapping)
        self.assertIn("t", ising_mapping)

        # Check that J and t have the correct types and shapes
        self.assertIsInstance(ising_mapping["J"], np.ndarray)
        self.assertIsInstance(ising_mapping["t"], np.ndarray)
        self.assertEqual(ising_mapping["J"].shape[0], ising_mapping["J"].shape[1], "J matrix should be square.")
        self.assertEqual(len(ising_mapping["t"]), ising_mapping["J"].shape[0],
                         "t vector length should match J matrix size.")

        self.assertGreater(mapping_time, 0, "Mapping time should be greater than zero.")

    def test_reverse_map(self):
        # Create a mock solution
        mock_solution = [1, 0, 1, 1, 0]

        # Run reverse_map to convert the solution back
        reverse_mapped_solution, reverse_mapping_time = self.choi_ising_instance.reverse_map(mock_solution)

        self.assertIsInstance(reverse_mapped_solution, dict)
        self.assertGreater(reverse_mapping_time, 0, "Reverse mapping time should be greater than zero.")

    def test_get_default_submodule(self):
        # Check QAOA submodule retrieval
        submodule = self.choi_ising_instance.get_default_submodule("QAOA")
        self.assertIsNotNone(submodule, "QAOA submodule should not be None")

        # Check PennylaneQAOA submodule retrieval
        submodule = self.choi_ising_instance.get_default_submodule("PennylaneQAOA")
        self.assertIsNotNone(submodule, "PennylaneQAOA submodule should not be None")

        # Check invalid option raises NotImplementedError
        with self.assertRaises(NotImplementedError):
            self.choi_ising_instance.get_default_submodule("InvalidSubmodule")
