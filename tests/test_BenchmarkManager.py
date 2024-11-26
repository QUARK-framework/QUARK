
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))



import unittest
from unittest.mock import MagicMock, patch, mock_open
import os
from pathlib import Path
from src.BenchmarkManager import BenchmarkManager, Instruction, preprocess, postprocess


class TestBenchmarkManager(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.benchmark_manager = BenchmarkManager()

    def test_initialization(self):
        # Reset store_dir before testing
        self.benchmark_manager.store_dir = None
        
        # Assertions
        self.assertIsNone(self.benchmark_manager.store_dir, "Expected store_dir to be None after initialization.")
        self.assertIsNone(self.benchmark_manager.application, "Expected application to be None after initialization.")
        self.assertEqual(self.benchmark_manager.results, [], "Expected results to be an empty list after initialization.")
        self.assertFalse(self.benchmark_manager.fail_fast, "Expected fail_fast to be False after initialization.")

    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data='[{"result": "data"}]')
    def test_load_interrupted_results(self, mock_open, mock_exists):
        mock_exists.return_value = True
        self.benchmark_manager.interrupted_results_path = "/tmp/results.json"
        results = self.benchmark_manager.load_interrupted_results()
        self.assertEqual(results, [{"result": "data"}])
        mock_open.assert_called_with("/tmp/results.json", encoding="utf-8")

    @patch("os.path.exists")
    def test_load_interrupted_results_no_file(self, mock_exists):
        mock_exists.return_value = False
        self.benchmark_manager.interrupted_results_path = "/tmp/results.json"
        results = self.benchmark_manager.load_interrupted_results()
        self.assertIsNone(results)

    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists", return_value=False)  # Mock `exists` to ensure the directory is considered non-existent
    def test_create_store_dir(self, mock_exists, mock_mkdir):
        with patch.object(self.benchmark_manager, "_set_logger") as mock_logger:
            self.benchmark_manager._create_store_dir(store_dir="/tmp", tag="test")
            # Assertions
            self.assertTrue(self.benchmark_manager.store_dir.startswith("/tmp/benchmark_runs/test-"))
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
            mock_logger.assert_called_once()

    @patch("logging.FileHandler")
    def test_resume_store_dir(self, mock_file_handler):
        # Ensure the directory exists
        test_dir = "/tmp/existing_dir"
        os.makedirs(test_dir, exist_ok=True)

        try:
            self.benchmark_manager._resume_store_dir(store_dir=test_dir)

            # Assertions
            self.assertEqual(self.benchmark_manager.store_dir, test_dir)
            mock_file_handler.assert_called_once_with(f"{test_dir}/logging.log")

        finally:
            # Cleanup the created directory
            if os.path.exists(test_dir):
                os.rmdir(test_dir)

    @patch("logging.getLogger")
    def test_set_logger(self, mock_get_logger):
        self.benchmark_manager.store_dir = "/tmp"
        self.benchmark_manager._set_logger()
        mock_logger = mock_get_logger.return_value
        mock_logger.addHandler.assert_called_once()

    def test_prepend_instruction(self):
        # Create a mock module_instance
        mock_module_instance = MagicMock()
        mock_module_instance.preprocess.return_value = (Instruction.PROCEED, "some_result")

        # Call preprocess with the mock
        result = preprocess(mock_module_instance)

        # Assertions
        self.assertEqual(result[0], Instruction.PROCEED, "Expected the first item to be Instruction.PROCEED.")
        self.assertEqual(result[1], "some_result", "Expected the second item to match the mock return")

    @patch("src.BenchmarkManager.Core")
    def test_postprocess(self, mock_core):
        mock_instance = MagicMock()
        mock_instance.postprocess.return_value = ("data",)
        result = postprocess(mock_instance)
        self.assertEqual(result[0], Instruction.PROCEED)

    @patch("BenchmarkManager.comm")
    @patch("BenchmarkManager.ConfigManager")
    def test_orchestrate_benchmark(self, mock_config_manager, mock_comm):
        mock_comm.Get_rank.return_value = 0  # Simulate master node in MPI
        mock_config_manager_instance = MagicMock()
        mock_config_manager.return_value = mock_config_manager_instance
        mock_config_manager_instance.get_config.return_value = {"application": {"name": "TestApp"}}
        mock_config_manager_instance.get_reps.return_value = 2
        mock_config_manager_instance.start_create_benchmark_backlog.return_value = [{"config": {}, "submodule": None}]
        mock_config_manager_instance.get_app.return_value = MagicMock()

        manager = BenchmarkManager()
        manager._save_as_json = MagicMock()  # Mock result saving
        manager._collect_all_results = MagicMock(return_value=[])
        manager._create_store_dir = MagicMock()  # Mock directory creation

        manager.orchestrate_benchmark(mock_config_manager_instance, [])

        # Assertions
        manager._create_store_dir.assert_called_once()
        mock_config_manager_instance.start_create_benchmark_backlog.assert_called_once()
        manager._collect_all_results.assert_called_once()
        manager._save_as_json.assert_called_once()
