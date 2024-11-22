import unittest
import os
import time
import nnf
from tempfile import TemporaryDirectory

from modules.applications.optimization.SAT.SAT import SAT


class TestSAT(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sat_instance = SAT()
        cls.problem = cls.sat_instance.generate_problem(
            {"variables": 10, "clvar_ratio_cons": 2, "clvar_ratio_test": 2, "problem_set": 5, "max_tries": 100})
        
    @classmethod
    def tearDownClass(cls):
        del cls.sat_instance
    
    def test_get_requirements(self):
        requirements = self.sat_instance.get_requirements()
        self.assertIn({"name": "nnf", "version": "0.4.1"}, requirements)
        self.assertIn({"name": "numpy", "version": "1.26.4"}, requirements)

    def test_get_default_submodule(self):
        submodules = [
            "QubovertQUBO", "Direct", "ChoiQUBO", "ChoiIsing", "DinneenQUBO", "DinneenIsing"
        ]
        for option in submodules:
            with self.subTest(option=option):
                submodule = self.sat_instance.get_default_submodule(option)
                self.assertIsNotNone(submodule, f"{option} submodule should not be None")

        with self.assertRaises(NotImplementedError):
            self.sat_instance.get_default_submodule("DirectX")

    def test_get_parameter_options(self):
        options = self.sat_instance.get_parameter_options()
        self.assertIn("variables", options)
        self.assertIn("clvar_ratio_cons", options)

    def test_generate_problem(self):
        # Define a configuration for testing
        config = {
            "variables": 10,
            "clvar_ratio_cons": 2,
            "clvar_ratio_test": 2,
            "problem_set": 1,
            "max_tries": 5000
        }

        # Generate the problem
        hard, soft = self.sat_instance.generate_problem(config)

        # Check the types of the output
        self.assertIsInstance(hard, nnf.And, "Hard constraints should be of type nnf.And.")
        self.assertIsInstance(soft, list, "Soft constraints should be a list.")

        # Validate the structure and cardinalities
        hard_clauses = list(hard)  # Extract the list of clauses from the And object
        self.assertEqual(len(hard_clauses), round(config["clvar_ratio_cons"] * config["variables"]),
                        "Incorrect number of hard constraints.")
        self.assertEqual(len(soft), round(config["clvar_ratio_test"] * config["variables"]),
                        "Incorrect number of soft constraints.")

    def test_validate(self):
        self.sat_instance.generate_problem(
            {"variables": 10, "clvar_ratio_cons": 2, "clvar_ratio_test": 2, "problem_set": 5, "max_tries": 100}
        )
        right_solution = {'L5': 1, 'L8': 0, 'L7': 0, 'L1': 1, 'L4': 1, 'L0': 1, 'L9': 1, 'L3': 1, 'L6': 1, 'L2': 1}
        self.assertEqual(self.sat_instance.validate(right_solution)[0], True)

        wrong_solution = right_solution.copy()
        wrong_solution["L5"] = 1 - wrong_solution["L5"]
        self.assertEqual(self.sat_instance.validate(wrong_solution)[0], False)

        # Edge Case: Empty solution
        empty_solution = {}
        with self.assertRaises(ValueError):
            self.sat_instance.validate(empty_solution)

        # Edge Case: Partial solution
        partial_solution = {'L1': 1, 'L2': 0}
        with self.assertRaises(ValueError):
            self.sat_instance.validate(partial_solution)
    
    def test_evaluate(self):
        solution = {'L5': 1, 'L8': 0, 'L7': 0, 'L1': 1, 'L4': 1, 'L0': 1, 'L9': 1, 'L3': 1, 'L6': 1, 'L2': 1}
        self.assertAlmostEqual(self.sat_instance.evaluate(solution)[0], 0.95, delta=0.05)

        # Edge Case: All True
        all_true_solution = {f'L{i}': 1 for i in range(10)}
        self.assertGreaterEqual(self.sat_instance.evaluate(all_true_solution)[0], 0)

        # Edge Case: All False
        all_false_solution = {f'L{i}': 0 for i in range(10)}
        self.assertLessEqual(self.sat_instance.evaluate(all_false_solution)[0], 1)

    def test_save(self):
        with TemporaryDirectory() as temp_dir:
            self.sat_instance.save(temp_dir, 1)

            file_paths = [f"{temp_dir}/constraints_iter_1.cnf", f"{temp_dir}/tests_iter_1.cnf"]

            for file_path in file_paths:
                self.assertTrue(os.path.isfile(file_path))

                file_extension = os.path.splitext(file_path)[-1].lower()
                self.assertEqual(file_extension, ".cnf")

                file_length = os.stat(file_path).st_size == 0
                self.assertFalse(file_length)
    
