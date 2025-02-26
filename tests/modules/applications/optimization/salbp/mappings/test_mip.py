import unittest
from unittest.mock import patch, MagicMock
from docplex.mp.model import Model
from docplex.mp.dvar import Var
from modules.applications.optimization.salbp.mappings.mip import MIP
from modules.applications.optimization.salbp.salbp import Task, SALBPInstance


class TestMIP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mip_instance = MIP()
        cls.tasks = frozenset({Task(1, 3), Task(2, 2), Task(3, 1)})
        cls.precedences = frozenset({(Task(1, 3), Task(2, 2))})
        cls.salbp_instance = SALBPInstance(cycle_time=5, tasks=cls.tasks, preceding_tasks=cls.precedences)

    def test_initialization(self):
        self.assertListEqual(self.mip_instance.submodule_options, ["MIPSolver"])
        self.assertIsNone(self.mip_instance.salbp, "Expected salbp to be None.")
        self.assertIsNone(self.mip_instance.config, "Expected config to be None.")

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
        config = {}
        with patch("modules.applications.optimization.salbp.mappings.mip.Model", return_value=Model()):
            model, processing_time = self.mip_instance.map(self.salbp_instance, config)

        self.assertIsInstance(model, Model, "Expected output to be an instance of Model.")
        self.assertIsInstance(processing_time, float, "Expected processing_time to be a float.")

    def test_add_variables(self):
        station_vars, task_station_vars = self.mip_instance._add_variables(self.salbp_instance, Model(), 3)

        self.assertIsInstance(station_vars, list, "Expected station variables to be a list.")
        self.assertIsInstance(task_station_vars, dict, "Expected task-station variables to be a dictionary.")
        self.assertGreater(len(station_vars), 0, "Expected at least one station variable.")
        self.assertGreater(len(task_station_vars), 0, "Expected task-station assignments to exist.")

    def test_add_one_station_per_task_constraints(self):
        mock_model = MagicMock()
        task_station_vars = {(t.id, s): MagicMock() for t in self.tasks for s in range(3)}

        self.mip_instance._add_one_station_per_task_constraints(mock_model, self.tasks, 3, task_station_vars)
        mock_model.add_constraints.assert_called()

    def test_add_cycle_time_constraints(self):
        mock_model = MagicMock()
        task_station_vars = {(t.id, s): MagicMock() for t in self.tasks for s in range(3)}
        station_vars = [MagicMock() for _ in range(3)]

        self.mip_instance._add_cycle_time_constraints(mock_model, self.salbp_instance, 3, task_station_vars, station_vars)
        mock_model.add_constraints.assert_called()

    def test_add_preceding_tasks_constraints(self):
        mock_model = MagicMock()
        task_station_vars = {(t.id, s): MagicMock() for t in self.tasks for s in range(3)}

        self.mip_instance._add_preceding_tasks_constraints(mock_model, self.salbp_instance, 3, task_station_vars)
        mock_model.add_constraints.assert_called()

    def test_add_consecutive_stations_constraints(self):
        mock_model = MagicMock()
        station_vars = [MagicMock() for _ in range(3)]

        self.mip_instance._add_consecutive_stations_constraints(mock_model, 3, station_vars)
        mock_model.add_constraints.assert_called()

    def test_reverse_map(self):
        solution = {"y_1": 1, "y_2": 1, "x_1_1": 1, "x_2_2": 1}
        self.mip_instance.salbp = self.salbp_instance

        with patch.object(self.salbp_instance, "get_task", side_effect=lambda x: Task(x, 2)):
            task_assignment, _ = self.mip_instance.reverse_map(solution)

        self.assertIsInstance(task_assignment, dict, "Expected task assignment to be a dictionary.")
        self.assertIn(1, task_assignment, "Expected station 1 in task assignment.")
        self.assertIn(2, task_assignment, "Expected station 2 in task assignment.")

    def test_get_default_submodule(self):
        submodule = self.mip_instance.get_default_submodule("MIPSolver")
        self.assertIsNotNone(submodule, "Expected 'MIPSolver' submodule to be returned.")

        with self.assertRaises(NotImplementedError):
            self.mip_instance.get_default_submodule("InvalidSubmodule")