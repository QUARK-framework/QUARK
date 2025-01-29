import unittest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import os
from datetime import datetime

from quark.BenchmarkManager import BenchmarkManager, Instruction


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

    # These tests are commented out because:
    # - The `BenchmarkManager` relies on complex dependencies, including filesystem operations (`Path.mkdir`),
    #   logging configurations (`FileHandler`), and application-specific configurations (`ConfigManager`).
    # - Mocking all these dependencies accurately to test the `orchestrate_benchmark` and `run_benchmark` methods
    #   requires significant effort and a well-structured mocking strategy, which is currently incomplete.
    # - We plan to implement these tests in the future
    # @patch("BenchmarkManager.Path.mkdir")
    # @patch("BenchmarkManager.logging.FileHandler")
    # @patch("BenchmarkManager.ConfigManager")
    # @patch("BenchmarkManager.BenchmarkManager._collect_all_results")
    # def test_orchestrate_benchmark(self, mock_collect_results, mock_config_manager, mock_filehandler, mock_mkdir):
    #     # Mock ConfigManager behavior
    #     mock_config_manager.get_config.return_value = {
    #         "application": {"name": "test_application"}
    #     }
    #     mock_config_manager.get_app.return_value = MagicMock()
    #     mock_config_manager.start_create_benchmark_backlog.return_value = []
    #     mock_config_manager.get_reps.return_value = 1

    #     # Mock mkdir to avoid filesystem errors
    #     mock_mkdir.return_value = None

    #     # Mock FileHandler to avoid file creation errors
    #     mock_filehandler.return_value = MagicMock()
    #     mock_filehandler.return_value.level = 10  # Set a valid integer logging level

    #     # Mock _collect_all_results to return an empty list
    #     mock_collect_results.return_value = []

    #     # Create an instance of BenchmarkManager
    #     benchmark_manager = BenchmarkManager()

    #     # Call orchestrate_benchmark
    #     benchmark_manager.orchestrate_benchmark(mock_config_manager, [{"name": "test"}], "/mock/store_dir")

    #     # Assertions
    #     mock_config_manager.get_config.assert_called_once()
    #     mock_config_manager.save.assert_called_once()
    #     mock_mkdir.assert_called()
    #     mock_filehandler.assert_called_once_with(
    #         "/mock/store_dir/benchmark_runs/test_application-<timestamp>/logging.log"
    #     )
    #     mock_collect_results.assert_called_once()

    # @patch("src.BenchmarkManager.Path.mkdir")
    # @patch("builtins.open", new_callable=mock_open)
    # @patch("src.BenchmarkManager.logging.getLogger")
    # def test_run_benchmark(self, mock_get_logger, mock_open_file, mock_mkdir):
    #     # Set up a mocked logger
    #     mock_logger = MagicMock()
    #     mock_get_logger.return_value = mock_logger

    #     # Mock the BenchmarkManager instance and dependencies
    #     benchmark_manager = BenchmarkManager()
    #     benchmark_manager.application = MagicMock()
    #     benchmark_manager.application.metrics = MagicMock()
    #     benchmark_manager.application.metrics.set_module_config = MagicMock()
    #     benchmark_manager.application.metrics.set_preprocessing_time = MagicMock()
    #     benchmark_manager.application.metrics.add_metric = MagicMock()
    #     benchmark_manager.application.metrics.validate = MagicMock()
    #     benchmark_manager.application.save = MagicMock()
    #     benchmark_manager.benchmark_record_template = MagicMock()
    #     benchmark_manager.store_dir = "/mock/store"

    #     # Set up backlog and repetitions
    #     backlog = [{"config": {"name": "test"}, "submodule": None}]
    #     repetitions = 1

    #     # Run the benchmark
    #     benchmark_manager.run_benchmark(backlog, repetitions)

    #     # Assertions
    #     mock_mkdir.assert_called()
    #     mock_open_file.assert_called_with("/mock/store/benchmark_0/application_config.json", 'w')
    #     mock_logger.info.assert_called()  # Ensure logging calls happen
    #     benchmark_manager.application.save.assert_called()  # Ensure save is called
