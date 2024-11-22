import unittest
import os
from tempfile import TemporaryDirectory

from networkx import Graph
from modules.applications.optimization.PVC.PVC import PVC


class TestPVC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pvc_instance = PVC()

    def setUp(self):
        # refresh the problem before each test to avoid interdependency
        self.pvc_instance.generate_problem({"seams": 3})

    @classmethod
    def tearDownClass(cls):
        del cls.pvc_instance

    def test_initialization(self):
        self.assertEqual(self.pvc_instance.name, "PVC")
        self.assertEqual(self.pvc_instance.submodule_options, [
            "Ising", "QUBO", "GreedyClassicalPVC", "ReverseGreedyClassicalPVC", "RandomPVC"
        ])

    def test_get_requirements(self):
        requirements = self.pvc_instance.get_requirements()
        expected_requirements = [
            {"name": "networkx", "version": "3.2.1"},
            {"name": "numpy", "version": "1.26.4"}
        ]
        self.assertEqual(requirements, expected_requirements)

    def test_get_solution_quality_unit(self):
        self.assertEqual(self.pvc_instance.get_solution_quality_unit(), "Tour cost")

    def test_get_default_submodule(self):
        submodules = ["Ising", "QUBO", "GreedyClassicalPVC", "ReverseGreedyClassicalPVC", "RandomPVC"]
        for option in submodules:
            with self.subTest(option=option):
                submodule = self.pvc_instance.get_default_submodule(option)
                self.assertIsNotNone(submodule, f"{option} submodule should not be None")

        with self.assertRaises(NotImplementedError):
            self.pvc_instance.get_default_submodule("DirectX")

    def test_get_parameter_options(self):
        parameter_options = self.pvc_instance.get_parameter_options()
        expected_options = {
            "seams": {
                "values": list(range(1, 18)),
                "description": "How many seams should be sealed?"
            }
        }
        self.assertEqual(parameter_options, expected_options, "Parameter options do not match expected structure.")

    def test_generate_problem(self):
        with self.subTest("Default Config (None)"):
            config = None
            self.assertIsInstance(self.pvc_instance.generate_problem(config), Graph)

        with self.subTest("Minimal Seams"):
            config = {"seams": 1}
            self.assertIsInstance(self.pvc_instance.generate_problem(config), Graph)

        with self.subTest("Max Seams"):
            config = {"seams": 17}
            graph = self.pvc_instance.generate_problem(config)
            self.assertIsInstance(graph, Graph)
            # Ensure the graph is not empty
            self.assertGreater(len(graph.nodes), 0, "Generated graph should have at least one node.")

    def test_process_solution(self):

        solution = {
            ((0, 0), 1, 1, 0): 1,
            ((1, 0), 1, 1, 1): 1,
            ((2, 0), 1, 1, 2): 1,
            ((3, 0), 1, 1, 3): 1
        }
        route, processing_time = self.pvc_instance.process_solution(solution)

        # Check that the route is processed correctly
        self.assertEqual(len(route), 4, "Expected 4 steps in the route.")
        self.assertGreater(processing_time, 0, "Processing time should be positive.")

    def test_validate(self):
        config = {"seams": 3}
        self.pvc_instance.generate_problem(config)

        # Construct a valid solution
        solution = [
            ((0, 0), 1, 1),  # Base node
            ((1, 0), 1, 1),  # Seam 1
            ((2, 0), 1, 1),  # Seam 2
            ((3, 0), 1, 1),  # Seam 3
        ]
        is_valid, validation_time = self.pvc_instance.validate(solution)

        # Check that the solution is validated correctly
        self.assertTrue(is_valid, "Expected the solution to be valid.")
        self.assertGreater(validation_time, 0, "Validation time should be positive.")

    def test_evaluate(self):
        solution = [((0, 0), 1, 1), ((2, 5), 1, 0), ((1, 7), 1, 0)]
        self.assertEqual(self.pvc_instance.evaluate(solution)[0], 13.54028)

    def test_save(self):
        with TemporaryDirectory() as temp_dir:
            self.pvc_instance.save(temp_dir, 1)

            file_path = f"{temp_dir}/graph_iter_1.gpickle"
            self.assertTrue(os.path.isfile(file_path), "The file was not created as expected.")

            file_extension = os.path.splitext(file_path)[-1].lower()
            self.assertEqual(file_extension, ".gpickle", "The file extension is incorrect.")
