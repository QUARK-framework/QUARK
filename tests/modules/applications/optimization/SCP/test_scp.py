import unittest
import os
import pickle
from tempfile import TemporaryDirectory

from modules.applications.optimization.scp.scp import SCP


class TestSCP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scp_instance = SCP()

    def test_get_solution_quality_unit(self):
        unit = self.scp_instance.get_solution_quality_unit()
        self.assertEqual(unit, "Number of selected subsets", "Incorrect solution quality unit.")

    def test_get_default_submodule(self):
        submodule = self.scp_instance.get_default_submodule("qubovertQUBO")
        self.assertIsNotNone(submodule, "Expected 'qubovertQUBO' submodule to be returned.")
        with self.assertRaises(NotImplementedError):
            self.scp_instance.get_default_submodule("InvalidOption")

    def test_get_parameter_options(self):
        options = self.scp_instance.get_parameter_options()
        self.assertIn("model_select", options)
        self.assertIn("values", options["model_select"])
        self.assertIn("description", options["model_select"])

    def test_generate_problem(self):
        # Test Tiny configuration
        elements, subsets = self.scp_instance.generate_problem({"model_select": "Tiny"})
        self.assertEqual(elements, {1, 2, 3})
        self.assertEqual(subsets, [{1, 2}, {1, 3}, {3, 4}])

        # Test Small configuration
        elements, subsets = self.scp_instance.generate_problem({"model_select": "Small"})
        self.assertEqual(elements, set(range(1, 15)))
        self.assertEqual(len(subsets), 8)

        # Test Large configuration
        with self.assertRaises(ValueError):
            self.scp_instance.generate_problem({"model_select": "InvalidModel"})

    def test_process_solution(self):
        # Setup for Tiny problem
        self.scp_instance.generate_problem({"model_select": "Tiny"})
        solution = [0, 1]
        processed_solution, processing_time = self.scp_instance.process_solution(solution)
        self.assertEqual(processed_solution, [[1, 2], [1, 3]])
        self.assertGreater(processing_time, 0, "Processing time should be positive.")

    def test_validate(self):
        # Setup for Tiny problem
        self.scp_instance.generate_problem({"model_select": "Tiny"})
        valid_solution = [[1, 2], [1, 3]]
        invalid_solution = [[1, 2]]

        # Valid solution
        is_valid, validation_time = self.scp_instance.validate(valid_solution)
        self.assertTrue(is_valid)
        self.assertGreater(validation_time, 0, "Validation time should be positive.")

        # Invalid solution
        is_valid, _ = self.scp_instance.validate(invalid_solution)
        self.assertFalse(is_valid, "Expected solution to be invalid.")

    def test_evaluate(self):
        solution = [[1, 2], [1, 3]]
        num_subsets, eval_time = self.scp_instance.evaluate(solution)
        self.assertEqual(num_subsets, len(solution))
        self.assertGreaterEqual(eval_time, 0, "Evaluation time should be positive.")

    def test_save(self):
        with TemporaryDirectory() as temp_dir:
            # Save the SCP instance
            self.scp_instance.generate_problem({"model_select": "Tiny"})
            self.scp_instance.save(temp_dir, iter_count=1)

            # Verify the saved file exists
            file_path = os.path.join(temp_dir, "SCP_instance")
            self.assertTrue(os.path.isfile(file_path), "SCP instance file was not saved.")

            # Check the file is readable and non-empty
            with open(file_path, "rb") as file:
                saved_data = pickle.load(file)
            self.assertIn("elements_to_cover", saved_data)
            self.assertIn("subsets", saved_data)
