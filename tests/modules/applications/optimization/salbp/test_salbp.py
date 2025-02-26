import unittest
from unittest.mock import patch, MagicMock
import networkx as nx
from pathlib import Path
from modules.applications.optimization.salbp.salbp import (
    Task, SALBPInstance, salbp_factory, has_overloaded_station, has_unique_assignment_for_every_task,
    respects_precedences, parse_task, create_salbp_from_file, SALBP
)

class TestSALBP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tasks = {Task(1, 3), Task(2, 2), Task(3, 1)}
        cls.precedences = {(Task(1, 3), Task(2, 2))}
        cls.salbp_instance = SALBPInstance(cycle_time=5, tasks=cls.tasks, preceding_tasks=cls.precedences)

    def test_task_creation(self):
        task = Task(1, 5)
        self.assertEqual(task.id, 1, "Expected task ID to be 1")
        self.assertEqual(task.time, 5, "Expected task time to be 5")

    def test_salbp_instance_creation(self):
        self.assertEqual(self.salbp_instance.cycle_time, 5, "Expected cycle time to be 5")
        self.assertEqual(len(self.salbp_instance.tasks), 3, "Expected 3 tasks in instance")
        self.assertEqual(len(self.salbp_instance.preceding_tasks), 1, "Expected 1 precedence constraint")

    def test_salbp_factory(self):
        instance = salbp_factory(list(self.tasks), list(self.precedences), 5)
        self.assertIsInstance(instance, SALBPInstance, "Expected instance to be of type SALBPInstance")

    def test_has_overloaded_station(self):
        task_assignment = {1: [Task(1, 3), Task(2, 4)]}
        self.assertTrue(has_overloaded_station(5, task_assignment), "Expected station to be overloaded")

    def test_has_unique_assignment_for_every_task(self):
        task_assignment = {
            1: [Task(1, 3)],
            2: [Task(2, 2)],
            3: [Task(3, 1)]
        }
        self.assertTrue(has_unique_assignment_for_every_task(self.tasks, task_assignment),
                        "Expected all tasks to be uniquely assigned")

    def test_respects_precedences(self):
        task_assignment = {1: [Task(1, 3)], 2: [Task(2, 2)]}
        self.assertTrue(respects_precedences(self.precedences, task_assignment),
                        "Expected precedences to be respected")

    def test_parse_task(self):
        task = parse_task("1 5")
        self.assertIsInstance(task, Task, "Expected a Task instance")
        self.assertEqual(task.id, 1, "Expected task ID to be 1")
        self.assertEqual(task.time, 5, "Expected task time to be 5")

    def test_create_salbp_from_file(self):
        mock_data = [
            "<number of tasks>", "3",
            "<cycle time>", "5",
            "<task times>", "1 3", "2 2", "3 1",
            "<precedence relations>",
            "1,2",
            "2,3",
            "<end>"
        ]

        mock_indices = {
            "<number of tasks>": 0,
            "<cycle time>": 2,
            "<task times>": 4,
            "<precedence relations>": 7,
            "<end>": 10
        }

        mock_split_lines = {
            "<number of tasks>": ["3"],
            "<cycle time>": ["5"],
            "<task times>": ["1 3", "2 2", "3 1"],
            "<precedence relations>": ["1,2", "2,3"],
        }

        with patch("modules.applications.optimization.salbp.salbp.read_data", return_value=mock_data):
            with patch("modules.applications.optimization.salbp.salbp.get_indices", return_value=mock_indices):
                with patch("modules.applications.optimization.salbp.salbp.split_lines_to_areas", return_value=mock_split_lines):
                    instance = create_salbp_from_file(Path("mock_path"))
                    self.assertIsInstance(instance, SALBPInstance, "Expected a valid SALBPInstance")

    def test_salbp_initialization(self):
        salbp = SALBP()
        self.assertEqual(salbp.name, "SALBP", "Expected name to be 'SALBP'")
        self.assertEqual(salbp.submodule_options, ["MIP"], "Expected submodule options to contain 'MIP'")

    def test_get_requirements(self):
        salbp = SALBP()
        requirements = salbp.get_requirements()
        expected_requirements = [
            {"name": "docplex", "version": "2.25.236"},
            {"name": "networkx", "version": "2.8.8"},
        ]
        self.assertEqual(requirements, expected_requirements, "Expected correct module dependencies")

    def test_generate_problem(self):
        salbp = SALBP()
        with patch("modules.applications.optimization.salbp.salbp.create_salbp_from_file", return_value=self.salbp_instance):
            instance = salbp.generate_problem({"instance": "example_instance_n=3.alb"})
            self.assertIsInstance(instance, SALBPInstance, "Expected a valid SALBPInstance")

    def test_validate_solution(self):
        salbp = SALBP()
        salbp.salbp = self.salbp_instance
        solution = {1: [Task(1, 3)], 2: [Task(2, 2)], 3: [Task(3, 1)]}

        with patch("modules.applications.optimization.salbp.salbp.has_overloaded_station", return_value=False):
            with patch("modules.applications.optimization.salbp.salbp.has_unique_assignment_for_every_task",
                       return_value=True):
                with patch("modules.applications.optimization.salbp.salbp.respects_precedences", return_value=True):
                    validity, _ = salbp.validate(solution)
                    self.assertTrue(validity, "Expected solution to be valid")

    def test_evaluate_solution(self):
        """
        Test evaluation function.
        """
        salbp = SALBP()
        salbp.salbp = self.salbp_instance
        solution = {1: [Task(1, 3)], 2: [Task(2, 2)], 3: [Task(3, 1)]}
        salbp.task_assignment = solution

        obj_value, _ = salbp.evaluate(solution)
        self.assertEqual(obj_value, 3, "Expected objective value to be 3 (number of used stations)")