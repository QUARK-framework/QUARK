import unittest
from unittest.mock import patch, MagicMock
from modules.applications.optimization.bp.mappings.qubo import QUBO


class TestQUBO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.qubo_instance = QUBO()

    def test_initialization(self):
        self.assertListEqual(self.qubo_instance.submodule_options, ["Annealer"])

    def test_get_requirements(self):
        requirements = self.qubo_instance.get_requirements()
        self.assertIsInstance(requirements, list, "Expected requirements to be a list.")
        self.assertEqual(len(requirements), 0, "Expected requirements list to be empty.")

    def test_get_parameter_options(self):
        params = self.qubo_instance.get_parameter_options()
        self.assertIn("penalty_factor", params, "Expected 'penalty_factor' in parameter options.")
        self.assertEqual(params["penalty_factor"]["values"], [1], "Expected penalty_factor values to be [1].")
        self.assertTrue(params["penalty_factor"]["custom_input"], "Expected custom_input to be True.")
        self.assertTrue(params["penalty_factor"]["allow_ranges"], "Expected allow_ranges to be True.")

    def test_map(self):
        config = {"penalty_factor": 2}
        problem = ([2, 4, 6], 10, [])

        with patch("modules.applications.optimization.bp.mappings.mip.MIP.create_MIP", return_value=MagicMock()):
            with patch("modules.applications.optimization.bp.mappings.qubo.QUBO.transform_docplex_mip_to_qubo",
                       return_value=(MagicMock(), MagicMock())):
                result, _ = self.qubo_instance.map(problem, config)

        self.assertIn("Q", result, "Expected 'Q' (QUBO operator) in result.")
        self.assertIn("QUBO", result, "Expected 'QUBO' representation in result.")
        self.assertIsNotNone(result["Q"], "Expected 'Q' to be non-empty.")
        self.assertIsNotNone(result["QUBO"], "Expected 'QUBO' to be non-empty.")

    def test_get_default_submodule(self):
        submodule = self.qubo_instance.get_default_submodule("Annealer")
        self.assertIsNotNone(submodule, "Expected 'Annealer' submodule to be returned.")

        with self.assertRaises(NotImplementedError):
            self.qubo_instance.get_default_submodule("InvalidSubmodule")
