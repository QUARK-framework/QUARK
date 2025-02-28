import os
import unittest
from tempfile import TemporaryDirectory
<<<<<<< HEAD:tests/modules/applications/optimization/acl/test_acl.py
from modules.applications.optimization.acl.acl import ACL
=======

import pandas as pd

from modules.applications.optimization.ACL.ACL import ACL
>>>>>>> GreshmaShaji-binpacking_and_mipsolver:tests/modules/applications/optimization/ACL/test_ACL.py


class TestACL(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.acl_instance = ACL()
        cls.config_tiny = {"model_select": "Tiny"}
        cls.config_small = {"model_select": "Small"}
        cls.config_full = {"model_select": "Full"}

        # Mock vehicle data
        cls.mock_data = pd.DataFrame({
            "Type": ["G20", "G07", "G20"],
            "Class": [1, 2, 1],
            "Length": [400, 500, 400],
            "Height": [140, 160, 140],
            "Weight": [15, 20, 15],
        })

        # Mock vehicles list
        cls.vehicles = ["G20", "G07", "G20"]

    def test_get_requirements(self):
        requirements = self.acl_instance.get_requirements()
        expected_requirements = [
            {"name": "pulp", "version": "2.9.0"},
            {"name": "pandas", "version": "2.2.3"},
            {"name": "numpy", "version": "1.26.4"},
            {"name": "openpyxl", "version": "3.1.5"},
        ]
        for req in expected_requirements:
            self.assertIn(req, requirements)

    def test_get_default_submodule(self):
        submodule = self.acl_instance.get_default_submodule("MIPsolverACL")
        self.assertIsNotNone(submodule, "Expected 'MIPsolverACL' submodule to be returned.")
        with self.assertRaises(NotImplementedError):
            self.acl_instance.get_default_submodule("InvalidSubmodule")

    def test_get_parameter_options(self):
        options = self.acl_instance.get_parameter_options()
        self.assertIn("model_select", options)
        self.assertIn("values", options["model_select"])
        self.assertIn("description", options["model_select"])

    def test_intersectset(self):
        set1 = [1, 2, 3]
        set2 = [2, 3, 4]
        expected_result = [2, 3]
        result = self.acl_instance.intersectset(set1, set2)
        self.assertEqual(result, expected_result, "Intersection result is incorrect.")

    def test_diffset(self):
        set1 = [1, 2, 3]
        set2 = [2, 3, 4]
        expected_result = [1]
        result = self.acl_instance.diffset(set1, set2)
        self.assertEqual(result, expected_result, "Difference result is incorrect.")

    def test_generate_problem(self):
        # Test Tiny configuration
        problem_tiny = self.acl_instance.generate_problem(self.config_tiny)
        self.assertIsInstance(problem_tiny, dict, "Expected Tiny problem to be a dictionary.")

        # Test Small configuration
        problem_small = self.acl_instance.generate_problem(self.config_small)
        self.assertIsInstance(problem_small, dict, "Expected Small problem to be a dictionary.")

        # Test Full configuration
        problem_full = self.acl_instance.generate_problem(self.config_full)
        self.assertIsInstance(problem_full, dict, "Expected Full problem to be a dictionary.")

    def test_get_solution_quality_unit(self):
        unit = self.acl_instance.get_solution_quality_unit()
        self.assertEqual(unit, "Number of loaded vehicles", "Solution quality unit mismatch.")

    def test_get_vehicle_params(self):
        class_list, length_list, height_list, weight_list = self.acl_instance._get_vehicle_params(
            self.mock_data, self.vehicles
        )
        self.assertEqual(class_list, [1, 2, 1], "Class list extraction failed.")
        self.assertEqual(length_list, [400, 500, 400], "Length list extraction failed.")
        self.assertEqual(height_list, [140, 160, 140], "Height list extraction failed.")
        self.assertEqual(weight_list, [15, 20, 15], "Weight list extraction failed.")

    def test_generate_full_model(self):
        self.acl_instance._generate_full_model(self.mock_data, self.vehicles)
        self.assertIsNotNone(self.acl_instance.application, "Full model generation failed.")
        self.assertTrue("objective" in self.acl_instance.application.to_dict(), "Full model lacks an objective.")

    def test_generate_small_model(self):
        self.acl_instance._generate_small_model(self.mock_data, self.vehicles)
        self.assertIsNotNone(self.acl_instance.application, "Small model generation failed.")
        self.assertTrue("objective" in self.acl_instance.application.to_dict(), "Small model lacks an objective.")

    def test_generate_tiny_model(self):
        self.acl_instance._generate_tiny_model(self.mock_data, self.vehicles)
        self.assertIsNotNone(self.acl_instance.application, "Tiny model generation failed.")
        self.assertTrue("objective" in self.acl_instance.application.to_dict(), "Tiny model lacks an objective.")

    def test_validate(self):
        # Create a mock solution
        mock_solution = {"status": "Optimal"}
        is_valid, _ = self.acl_instance.validate(mock_solution)
        self.assertTrue(is_valid, "Expected solution to be valid.")

        invalid_solution = {"status": "Infeasible"}
        is_valid, _ = self.acl_instance.validate(invalid_solution)
        self.assertFalse(is_valid, "Expected solution to be invalid.")

    def test_evaluate(self):
        # Create a mock solution with objective value and variable assignments
        mock_solution = {
            "obj_value": 5,
            "variables": {"x_0_1": 1, "x_1_2": 1, "x_2_3": 0}
        }
        obj_value, eval_time = self.acl_instance.evaluate(mock_solution)
        self.assertEqual(obj_value, 5, "Objective value is incorrect.")
        self.assertGreater(eval_time, 0, "Evaluation time should be positive.")

    def test_save(self):
        with TemporaryDirectory() as temp_dir:
            # Generate and save the problem instance
            self.acl_instance.generate_problem(self.config_tiny)
            self.acl_instance.save(temp_dir, iter_count=1)

            # Check that the file exists and is not empty
            file_path = os.path.join(temp_dir, "ACL_instance.json")
            self.assertTrue(os.path.isfile(file_path), "Problem instance file not saved.")
            self.assertTrue(file_path.endswith(".json"), "File extension mismatch.")
            self.assertGreater(os.path.getsize(file_path), 0, "Problem instance file is empty.")
