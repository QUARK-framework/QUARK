import unittest
import networkx as nx
import numpy as np
import os
from tempfile import TemporaryDirectory
import logging

from modules.applications.optimization.tsp.tsp import TSP


class TestTSP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tsp_instance = TSP()
        cls.config = {"nodes": 3}
        cls.graph = cls.tsp_instance.generate_problem(cls.config)

    @classmethod
    def tearDownClass(cls):
        del cls.tsp_instance

    def test_get_requirements(self):
        requirements = self.tsp_instance.get_requirements()
        self.assertIn({"name": "networkx", "version": "3.4.2"}, requirements)
        self.assertIn({"name": "numpy", "version": "1.26.4"}, requirements)

    def test_get_default_submodule(self):
        submodules = ["Ising", "QUBO", "GreedyClassicalTSP", "ReverseGreedyClassicalTSP", "RandomTSP"]
        for option in submodules:
            with self.subTest(option=option):
                submodule = self.tsp_instance.get_default_submodule(option)
                self.assertIsNotNone(submodule, f"{option} submodule should not be None")

        with self.assertRaises(NotImplementedError):
            self.tsp_instance.get_default_submodule("DirectX")

    def test_get_parameter_options(self):
        options = self.tsp_instance.get_parameter_options()
        self.assertIn("nodes", options)

    def test_generate_problem(self):
        graph = self.tsp_instance.generate_problem(self.config)
        self.assertIsInstance(graph, nx.Graph)
        self.assertEqual(len(graph.nodes), self.config["nodes"])

    def test_get_tsp_matrix(self):
        # Define a graph with nodes and edges
        graph = nx.Graph()
        graph.add_weighted_edges_from([
            (1, 2, 4),  # Edge (1, 2) with weight 4
            (1, 3, 7),  # Edge (1, 3) with weight 7
            (2, 3, 5),
        ])

        # Define the expected distance matrix
        expected_matrix = np.array([
            [0, 4, 7],  # Distances from node 0
            [4, 0, 5],  # Distances from node 1
            [7, 5, 0],  # Distances from node 2
        ])

        result_matrix = self.tsp_instance._get_tsp_matrix(graph)

        self.assertTrue(np.allclose(result_matrix, expected_matrix),
                        "The generated matrix does not match the expected matrix.")

    def test_process_solution(self):
        right_solution = {(0, 0): 1, (2, 1): 1, (1, 2): 1}
        right_route = [0, 2, 1]
        self.assertEqual(self.tsp_instance.process_solution(right_solution)[0], right_route)

        duplicate_node_solution = {(0, 0): 1, (0, 1): 1, (1, 2): 1}
        self.assertIsNone(self.tsp_instance.process_solution(duplicate_node_solution)[0])

        missing_nodes_solution = {(0, 0): 1, (1, 1): 1}
        self.assertIsNone(self.tsp_instance.process_solution(missing_nodes_solution)[0])

    def test_validate(self):
        logging.disable(logging.ERROR)
        # Valid solution
        valid_solution = list(self.graph.nodes)
        is_valid, validation_time = self.tsp_instance.validate(valid_solution)
        self.assertTrue(is_valid)
        self.assertGreater(validation_time, 0, "Validation time should be positive.")

        # Invalid solution
        invalid_solution = valid_solution[:-1]
        is_valid, _ = self.tsp_instance.validate(invalid_solution)
        self.assertFalse(is_valid)

    def test_evaluate(self):
        solution = list(self.graph.nodes)
        np.random.shuffle(solution)

        # Calculate the expected total distance
        total_distance = 0
        for i in range(len(solution) - 1):
            total_distance += self.graph[solution[i]][solution[i + 1]]["weight"]
        total_distance += self.graph[solution[-1]][solution[0]]["weight"]

        # Validate the evaluation function
        evaluated_distance, eval_time = self.tsp_instance.evaluate(solution)
        self.assertEqual(evaluated_distance, total_distance)
        self.assertGreater(eval_time, 0, "Evaluation time should be positive.")

    def test_save(self):
        with TemporaryDirectory() as temp_dir:
            # Save the generated graph
            self.tsp_instance.save(temp_dir, iter_count=1)

            # Check that the file exists and has the correct extension
            file_path = f"{temp_dir}/graph_iter_1.gpickle"
            self.assertTrue(os.path.isfile(file_path), "Expected file to be saved.")
            self.assertTrue(file_path.endswith(".gpickle"), "Expected file extension to be .gpickle")

            # Ensure file is non-empty
            self.assertGreater(os.path.getsize(file_path), 0, "Expected saved file to be non-empty.")
