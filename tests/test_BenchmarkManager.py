import os
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

from src.BenchmarkManager import BenchmarkManager, Instruction


class TestBenchmarkManager(unittest.TestCase):

    def setUp(self):
        """
        Set up resources before each test.
        """
        self.benchmark_manager = BenchmarkManager()
        self.benchmark_manager.store_dir = "/mock/store"
        self.benchmark_manager.application = MagicMock()
        self.benchmark_manager.application.metrics = MagicMock()

    def test_initialization(self):
        # Reset store_dir before testing
        self.benchmark_manager.store_dir = None

        # Assertions
        self.assertIsNone(self.benchmark_manager.store_dir, "Expected store_dir to be None after initialization.")
        self.assertEqual(
            self.benchmark_manager.results,
            [],
            "Expected results to be an empty list after initialization.")

    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data='[{"key": "value"}]')
    def test_load_interrupted_results(self, mock_open_file, mock_path_exists):
        self.benchmark_manager.interrupted_results_path = "mock_path"
        results = self.benchmark_manager.load_interrupted_results()
        mock_open_file.assert_called_with("mock_path", encoding="utf-8")
        self.assertEqual(results, [{"key": "value"}])

    @patch("os.path.exists", return_value=False)
    def test_load_interrupted_results_no_file(self, mock_path_exists):
        self.benchmark_manager.interrupted_results_path = "mock_path"
        results = self.benchmark_manager.load_interrupted_results()
        self.assertIsNone(results)

    @patch("BenchmarkManager.Path.mkdir")  # Mock Path.mkdir
    @patch("BenchmarkManager.logging.FileHandler")  # Mock FileHandler
    def test_create_store_dir(self, mock_file_handler, mock_path_mkdir):
        # Mock datetime to control the generated timestamp
        dynamic_now = datetime.today()
        expected_date_str = dynamic_now.strftime("%Y-%m-%d-%H-%M-%S")

        # Call the method under test
        self.benchmark_manager._create_store_dir(store_dir="/mock_dir", tag="test_tag")

        # Dynamically build the expected directory path
        expected_dir = f"/mock_dir/benchmark_runs/test_tag-{expected_date_str}"
        expected_log_file = f"{expected_dir}/logging.log"

        # Assertions to check expected outcomes
        self.assertEqual(self.benchmark_manager.store_dir, expected_dir)
        self.assertTrue(self.benchmark_manager.store_dir.startswith("/mock_dir/benchmark_runs/test_tag-"))
        self.assertTrue(self.benchmark_manager.store_dir.endswith(expected_date_str))
        mock_path_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_file_handler.assert_called_once_with(expected_log_file)

    @patch("logging.FileHandler")
    @patch("logging.getLogger")
    def test_set_logger(self, mock_get_logger, mock_file_handler):
        """
        Test _set_logger method.
        """
        self.benchmark_manager.store_dir = "/mock/store"
        logger_mock = MagicMock()
        mock_get_logger.return_value = logger_mock

        self.benchmark_manager._set_logger()

        mock_file_handler.assert_called_with("/mock/store/logging.log")
        logger_mock.addHandler.assert_called_once()

    @patch("BenchmarkManager.Path.mkdir")
    @patch("os.path.exists", return_value=True)
    @patch("BenchmarkManager.logging.FileHandler")
    def test_resume_store_dir(self, mock_file_handler, mock_path_exists, mock_path_mkdir):
        store_dir = "/mock_dir"
        self.benchmark_manager._resume_store_dir(store_dir)
        self.assertEqual(self.benchmark_manager.store_dir, store_dir)
        mock_file_handler.assert_called_once_with(f"{store_dir}/logging.log")

    @patch("glob.glob", return_value=["/mock_dir/results_1.json", "/mock_dir/results_2.json"])
    @patch("builtins.open", new_callable=mock_open, read_data='[{"result": "test1"}, {"result": "test2"}]')
    def test_collect_all_results(self, mock_open_file, mock_glob):
        results = self.benchmark_manager._collect_all_results()
        self.assertEqual(len(results), 4, "Expected to collect results from multiple files.")
        self.assertEqual(results[0]["result"], "test1", "Expected the first result to match test1.")
        self.assertEqual(results[2]["result"], "test1", "Expected the first result in the second file to match test1.")

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_save_as_json(self, mock_json_dump, mock_open_file):
        mock_results = [{"test": "result1"}, {"test": "result2"}]
        self.benchmark_manager._save_as_json(mock_results)
        mock_open_file.assert_called_once_with(f"{self.benchmark_manager.store_dir}/results.json", 'w')
        mock_json_dump.assert_called_once_with(mock_results, mock_open_file(), indent=2)

    @patch("src.BenchmarkManager.Plotter.visualize_results")
    @patch("src.BenchmarkManager.BenchmarkManager._save_as_json")
    @patch("src.BenchmarkManager.ConfigManager")
    def test_summarize_results(self, mock_config_manager, mock_save_json, mock_visualize):
        self.benchmark_manager.summarize_results(["/mock/dir1", "/mock/dir2"])
        mock_save_json.assert_called()
        mock_visualize.assert_called()

    @patch("glob.glob", return_value=["/mock_dir1/results.json", "/mock_dir2/results.json"])
    @patch("builtins.open", new_callable=mock_open, read_data='[{"result": "mock"}]')
    def test_load_results(self, mock_open_file, mock_glob):
        results = self.benchmark_manager.load_results(["/mock_dir1", "/mock_dir2"])
        self.assertEqual(len(results), 4, "Expected to load results from both directories.")
        self.assertEqual(results[0]["result"], "mock", "Expected the first result to match mock.")

    @patch("src.BenchmarkManager.preprocess")
    @patch("src.BenchmarkManager.postprocess")
    def test_traverse_config(self, mock_postprocess, mock_preprocess):
        # Mock the preprocess function to return expected values
        mock_preprocess.return_value = (Instruction.PROCEED, "processed_input", 0.1)

        mock_postprocess.return_value = (Instruction.PROCEED, "postprocessed_output", 0.2)

        mock_benchmark_record_template = MagicMock()
        mock_benchmark_record_template.copy.return_value = MagicMock()
        self.benchmark_manager.benchmark_record_template = mock_benchmark_record_template
        module = {
            "TestModule": {
                "name": "TestModule",
                "config": {"key": "value"},
                "instance": MagicMock(),
                "submodule": {}
            }
        }
        module["TestModule"]["instance"].metrics = MagicMock()

        instruction, output, benchmark_record = self.benchmark_manager.traverse_config(
            module, "input_data", "/mock/path", 1
        )

        # Assertions
        self.assertEqual(instruction, Instruction.PROCEED, "Expected Instruction.PROCEED to be returned.")
        self.assertEqual(output, "postprocessed_output", "Expected processed output to match mock postprocess return.")
        self.assertIsNotNone(benchmark_record, "Expected a BenchmarkRecord instance.")

    @patch("BenchmarkManager.BenchmarkManager._collect_all_results", return_value=[{"key": "value"}])
    @patch("BenchmarkManager.BenchmarkManager._save_as_json")
    def test_orchestrate_benchmark(self, mock_save_as_json, mock_collect_all_results):
        mock_config_manager = MagicMock()
        mock_config_manager.get_config.return_value = {"application": {"name": "test_app"}}
        mock_config_manager.get_reps.return_value = 1

        self.benchmark_manager.orchestrate_benchmark(mock_config_manager, app_modules=[], store_dir="/tmp")

        mock_config_manager.save.assert_called_once_with(self.benchmark_manager.store_dir)
        mock_config_manager.load_config.assert_called_once_with([])
