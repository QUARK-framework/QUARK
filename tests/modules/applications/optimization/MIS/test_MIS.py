import unittest
import networkx as nx
import os
from tempfile import TemporaryDirectory
import logging

from modules.applications.optimization.MIS.MIS import MIS


class TestMIS(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mis_instance = MIS()
        cls.config = {"size": 5, "spacing": 0.5, "filling_fraction": 0.5}
        cls.graph = cls.mis_instance.generate_problem(cls.config)

    @classmethod
    def tearDownClass(cls):
        del cls.mis_instance

    def test_get_solution_quality_unit(self):
        unit = self.mis_instance.get_solution_quality_unit()
        self.assertEqual(unit, "Set size", "Incorrect solution quality unit.")

    def test_get_default_submodule(self):
        submodule = self.mis_instance.get_default_submodule("NeutralAtom")
        self.assertIsNotNone(submodule, "Expected 'NeutralAtom' submodule to be returned.")
        with self.assertRaises(NotImplementedError):
            self.mis_instance.get_default_submodule("InvalidOption")

    def test_get_parameter_options(self):
        options = self.mis_instance.get_parameter_options()
        self.assertIn("size", options)
        self.assertIn("spacing", options)
        self.assertIn("filling_fraction", options)

    def test_generate_problem(self):
        # Generate with valid configuration
        graph = self.mis_instance.generate_problem(self.config)
        self.assertIsInstance(graph, nx.Graph)
        self.assertGreaterEqual(len(graph.nodes), 1, "Graph should have at least 1 node.")
        self.assertGreaterEqual(len(graph.edges), 0, "Graph should have non-negative edges.")

        # Invalid configuration
        with self.assertRaises(AssertionError):
            self.mis_instance.generate_problem({"size": 5})  # Missing required parameters

    def test_process_solution(self):
        solution = [1, 3]
        processed_solution, processing_time = self.mis_instance.process_solution(solution)
        self.assertEqual(processed_solution, solution, "Processed solution does not match input.")
        self.assertGreaterEqual(processing_time, 0, "Processing time should be positive.")

    def test_validate(self):
        logging.disable(logging.WARNING)
        valid_solution = [3]
        is_valid, validation_time = self.mis_instance.validate(valid_solution)
        self.assertTrue(is_valid, "Expected valid solution.")
        self.assertGreater(validation_time, 0, "Validation time should be positive.")

        invalid_solution = [1, 2]
        is_valid, _ = self.mis_instance.validate(invalid_solution)
        self.assertFalse(is_valid, "Expected invalid solution.")

    def test_evaluate(self):
        solution = list(self.graph.nodes)[:3]
        set_size, eval_time = self.mis_instance.evaluate(solution)
        self.assertEqual(set_size, len(solution), "Set size mismatch.")
        self.assertGreater(eval_time, 0, "Evaluation time should be positive.")

    def test_save(self):
        with TemporaryDirectory() as temp_dir:
            # Save the graph
            self.mis_instance.save(temp_dir, iter_count=1)

            # Check that the file exists and is non-empty
            file_path = f"{temp_dir}/graph_iter_1.gpickle"
            self.assertTrue(os.path.isfile(file_path), "Graph file not saved.")
            self.assertTrue(file_path.endswith(".gpickle"), "File extension mismatch.")
            self.assertGreater(os.path.getsize(file_path), 0, "Graph file is empty.")
