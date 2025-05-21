import unittest

import numpy as np
from nnf import And, Or, Var

from modules.applications.optimization.sat.mappings.dinneenising import DinneenIsing


class TestDinneenIsing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dinneen_ising_instance = DinneenIsing()
        # Define a SAT problem with both hard and soft constraints
        hard_constraints = And([Or([Var("L1"), ~Var("L2")]), Or([~Var("L3"), Var("L4")])])
        soft_constraints = [Or([Var("L5"), ~Var("L6")])]
        cls.problem = (hard_constraints, soft_constraints)
        cls.config = {"lagrange": 1.0}

    def test_get_requirements(self):
        requirements = self.dinneen_ising_instance.get_requirements()
        self.assertIn({"name": "nnf", "version": "0.4.1"}, requirements)
        self.assertIn({"name": "numpy", "version": "1.26.4"}, requirements)
        self.assertIn({"name": "dimod", "version": "0.12.18"}, requirements)

    def test_get_parameter_options(self):
        options = self.dinneen_ising_instance.get_parameter_options()
        self.assertIn("lagrange", options)

    def test_map(self):
        ising_mapping, mapping_time = self.dinneen_ising_instance.map(self.problem, self.config)

        # Check that mapping results contain the expected "J" and "t" keys
        self.assertIn("J", ising_mapping)
        self.assertIn("t", ising_mapping)

        # Check that J and t are numpy arrays
        j_matrix = ising_mapping["J"]
        t_vector = ising_mapping["t"]
        self.assertIsInstance(j_matrix, np.ndarray)
        self.assertIsInstance(t_vector, np.ndarray)
        self.assertEqual(j_matrix.shape[0], j_matrix.shape[1], "J matrix should be square.")
        self.assertEqual(len(t_vector), j_matrix.shape[0], "t vector length should match J matrix size.")

        self.assertGreater(mapping_time, 0, "Mapping time should be greater than zero.")

    def test_reverse_map(self):
        mock_solution = {i: 1 if i % 2 == 0 else 0 for i in range(len(self.dinneen_ising_instance.problem[0].vars()))}

        reverse_mapped_solution, reverse_mapping_time = self.dinneen_ising_instance.reverse_map(mock_solution)

        # Verify that the output of reverse_map is a dictionary
        self.assertIsInstance(reverse_mapped_solution, dict)
        self.assertGreater(reverse_mapping_time, 0, "Reverse mapping time should be greater than zero.")

    def test_get_default_submodule(self):
        submodule = self.dinneen_ising_instance.get_default_submodule("QAOA")
        self.assertIsNotNone(submodule, "QAOA submodule should not be None")

        submodule = self.dinneen_ising_instance.get_default_submodule("PennylaneQAOA")
        self.assertIsNotNone(submodule, "PennylaneQAOA submodule should not be None")

        with self.assertRaises(NotImplementedError):
            self.dinneen_ising_instance.get_default_submodule("InvalidSubmodule")
