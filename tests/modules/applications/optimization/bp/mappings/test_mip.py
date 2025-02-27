import unittest
from unittest.mock import patch, MagicMock
from docplex.mp.model import Model
from modules.applications.optimization.bp.mappings.mip import MIP


class TestMIP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mip_instance = MIP()

    def test_initialization(self):
        self.assertListEqual(self.mip_instance.submodule_options, ["MIPSolver"])
        self.assertIsNone(self.mip_instance.key_mapping)
        self.assertIsNone(self.mip_instance.graph)
        self.assertIsNone(self.mip_instance.config)

    def test_get_requirements(self):
        requirements = self.mip_instance.get_requirements()
        self.assertIsInstance(requirements, list, "Expected requirements to be a list.")
        self.assertEqual(requirements[0]["name"], "docplex", "Expected first requirement to be 'docplex'.")
        self.assertEqual(requirements[0]["version"], "2.25.236", "Expected docplex version to be '2.25.236'.")

    def test_get_parameter_options(self):
        params = self.mip_instance.get_parameter_options()
        self.assertIsInstance(params, dict, "Expected parameter options to be a dictionary.")
        self.assertEqual(len(params), 0, "Expected parameter options to be an empty dictionary.")

    def test_map(self):
        config = {"modelling_goal": 1.0}
        problem = ([2, 4, 6], 10, [])

        with patch("modules.applications.optimization.bp.mappings.mip.MIP.create_MIP", return_value=Model()):
            model, processing_time = self.mip_instance.map(problem, config)

        self.assertIsInstance(model, Model, "Expected output to be an instance of Model.")
        self.assertIsInstance(processing_time, float, "Expected processing_time to be a float.")

    def test_get_default_submodule(self):
        submodule = self.mip_instance.get_default_submodule("MIPSolver")
        self.assertIsNotNone(submodule, "Expected 'MIPSolver' submodule to be returned.")

        with self.assertRaises(NotImplementedError):
            self.mip_instance.get_default_submodule("InvalidSubmodule")
