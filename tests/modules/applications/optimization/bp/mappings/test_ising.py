import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from modules.applications.optimization.bp.mappings.ising import Ising


class TestIsing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ising_instance = Ising()

    def test_initialization(self):
        self.assertListEqual(self.ising_instance.submodule_options, ["QAOA", "PennylaneQAOA", "QiskitQAOA"])
        self.assertIsNone(self.ising_instance.key_mapping)
        self.assertIsNone(self.ising_instance.graph)
        self.assertIsNone(self.ising_instance.config)

    def test_get_requirements(self):
        requirements = self.ising_instance.get_requirements()
        expected_requirements = [{"name": "numpy", "version": "1.26.4"}, {"name": "docplex", "version": "2.25.236"}]
        self.assertEqual(requirements, expected_requirements)

    def test_get_parameter_options(self):
        params = self.ising_instance.get_parameter_options()
        self.assertIn("penalty_factor", params)
        self.assertEqual(params["penalty_factor"]["values"], [2])
        self.assertIsInstance(params["penalty_factor"]["values"], list)

    def test_map(self):
        """
        Test mapping function for bin-packing to Ising formulation.
        """
        config = {"penalty_factor": 2}
        problem = ([2, 4, 6], 10, [])
        with patch("modules.applications.optimization.bp.mappings.mip.MIP.create_mip", return_value=MagicMock()):
            with patch("modules.applications.optimization.bp.mappings.ising.Ising.transform_docplex_mip_to_ising", return_value=(np.array([[1, -1], [-1, 1]]), np.array([1, -1]), 0, MagicMock())):
                result, _ = self.ising_instance.map(problem, config)

        self.assertIn("J", result)
        self.assertIn("t", result)
        self.assertIn("c", result)
        self.assertIn("QUBO", result)
        self.assertIsInstance(result["J"], np.ndarray)
        self.assertIsInstance(result["t"], np.ndarray)
        self.assertIsInstance(result["c"], (float, int))

    def test_reverse_map(self):
        solution = np.array([1, 0, -1])
        mock_qubo = MagicMock()
        mock_qubo.variables = [MagicMock(name=f"x{i}") for i in range(3)]
        self.ising_instance.qubo = mock_qubo  # Mock the QUBO instance

        result, _ = self.ising_instance.reverse_map(solution)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 3)

    def test_convert_ising_to_qubo(self):
        solution = np.array([1, 0, -1])
        qubo_solution = self.ising_instance._convert_ising_to_qubo(solution)
        self.assertIsInstance(qubo_solution, np.ndarray)
        self.assertEqual(len(qubo_solution), len(solution))
        self.assertTrue(all(x in [0, 1] for x in qubo_solution))

    def test_get_default_submodule(self):
        submodule = self.ising_instance.get_default_submodule("QAOA")
        self.assertIsNotNone(submodule, "Expected 'QAOA' submodule to be returned.")

        submodule = self.ising_instance.get_default_submodule("PennylaneQAOA")
        self.assertIsNotNone(submodule, "Expected 'PennylaneQAOA' submodule to be returned.")

        submodule = self.ising_instance.get_default_submodule("QiskitQAOA")
        self.assertIsNotNone(submodule, "Expected 'QiskitQAOA' submodule to be returned.")

        with self.assertRaises(NotImplementedError):
            self.ising_instance.get_default_submodule("InvalidSubmodule")
