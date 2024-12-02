import unittest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import os
import json
import logging
import tempfile
from datetime import datetime

from src.BenchmarkManager import BenchmarkManager, Instruction, JobStatus, preprocess, postprocess
from modules.Core import Core


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

    # @patch("BenchmarkManager.Plotter.visualize_results")
    # @patch("BenchmarkManager.BenchmarkManager.load_results", return_value=[{"result": "test"}])
    # @patch("BenchmarkManager.BenchmarkManager._create_store_dir")
    # @patch("BenchmarkManager.BenchmarkManager._save_as_json")
    # def test_summarize_results(self, mock_save_as_json, mock_create_store_dir, mock_load_results, mock_visualize):
    #     self.benchmark_manager.summarize_results(["/mock_dir1", "/mock_dir2"])
    #     mock_create_store_dir.assert_called_once_with(tag="summary")
    #     mock_load_results.assert_called_once_with(["/mock_dir1", "/mock_dir2"])
    #     mock_save_as_json.assert_called_once_with([{"result": "test"}])
    #     mock_visualize.assert_called_once_with([{"result": "test"}], self.benchmark_manager.store_dir)

    @patch("glob.glob", return_value=["/mock_dir1/results.json", "/mock_dir2/results.json"])
    @patch("builtins.open", new_callable=mock_open, read_data='[{"result": "mock"}]')
    def test_load_results(self, mock_open_file, mock_glob):
        results = self.benchmark_manager.load_results(["/mock_dir1", "/mock_dir2"])
        self.assertEqual(len(results), 4, "Expected to load results from both directories.")
        self.assertEqual(results[0]["result"], "mock", "Expected the first result to match mock.")

    # @patch("BenchmarkManager.preprocess", return_value=(Instruction.PROCEED, "processed_input", 0.1))
    # @patch("BenchmarkManager.postprocess", return_value=(Instruction.PROCEED, "postprocessed_input", 0.2))
    # @patch("BenchmarkRecord.BenchmarkRecord.append_module_record_left")
    # def test_traverse_config(self, mock_append_record, mock_postprocess, mock_preprocess):
    #     module = {
    #         "module1": {
    #             "name": "module1",
    #             "instance": MagicMock(),
    #             "config": {"param": "value"},
    #             "submodule": None,
    #         }
    #     }
    #     module_instance = module["module1"]["instance"]
    #     module_instance.metrics = MagicMock()

    #     instruction, output, benchmark_record = self.benchmark_manager.traverse_config(
    #         module, "input_data", "/mock_path", 1
    #     )

    #     # Assertions
    #     self.assertEqual(instruction, Instruction.PROCEED, "Expected instruction to be PROCEED.")
    #     self.assertEqual(output, "postprocessed_input", "Expected postprocessed input as output.")
    #     mock_preprocess.assert_called_once_with(
    #         module_instance, "input_data", {"param": "value"},
    #         store_dir="/mock_path", rep_count=1, previous_job_info=None
    #     )
    #     mock_postprocess.assert_called_once_with(
    #         module_instance, "processed_input", {"param": "value"},
    #         store_dir="/mock_path", rep_count=1, previous_job_info=None
    #     )
    #     mock_append_record.assert_called_once_with(module_instance.metrics)

    # @patch("BenchmarkManager.BenchmarkManager._save_as_json")
    # @patch("BenchmarkManager.BenchmarkManager._collect_all_results")
    # @patch("BenchmarkManager.BenchmarkManager._resume_store_dir")
    # @patch("BenchmarkManager.BenchmarkManager._create_store_dir")
    # @patch("BenchmarkManager.ConfigManager")
    # def test_orchestrate_benchmark(self, mock_config_manager, mock_create_store_dir, mock_resume_store_dir, mock_collect_all_results, mock_save_as_json):
    #     mock_config = MagicMock()
    #     mock_config.get_config.return_value = {"application": {"name": "test"}}
    #     mock_config_manager.return_value = mock_config

    #     # Mock `_collect_all_results` to return a non-empty list
    #     mock_collect_all_results.return_value = [{"mock_key": "mock_value"}]

    #     # Mock `_save_as_json` to do nothing
    #     mock_save_as_json.return_value = None

    #     # Use a temporary directory for the store
    #     with tempfile.TemporaryDirectory() as temp_store_dir:
    #         self.benchmark_manager.orchestrate_benchmark(mock_config_manager, [], temp_store_dir, None)

    #         # Assertions to validate expected calls
    #         mock_create_store_dir.assert_called_with(temp_store_dir, tag="test")
    #         mock_resume_store_dir.assert_not_called()
    #         mock_collect_all_results.assert_called_once()
    #         mock_save_as_json.assert_called_once_with([{"mock_key": "mock_value"}])

    # @patch("pathlib.Path.mkdir")
    # @patch("builtins.open", new_callable=MagicMock)
    # @patch("BenchmarkManager.BenchmarkRecord")
    # @patch("BenchmarkManager.get_git_revision", return_value=("mock_git_revision", False))
    # @patch("BenchmarkManager.preprocess", return_value=(Instruction.PROCEED, {"mock_problem": "value"}, 1.0))
    # @patch("BenchmarkManager.postprocess", return_value=(Instruction.PROCEED, {}, 1.0))
    # @patch("BenchmarkManager.BenchmarkManager.load_interrupted_results", return_value=None)
    # @patch("utils_mpi.get_comm")
    # def test_run_benchmark_basic(
    #     self, mock_comm, mock_load_results, mock_postprocess, mock_preprocess,
    #     mock_git_revision, mock_benchmark_record, mock_open, mock_mkdir
    # ):
    #     """
    #     Test run_benchmark method with a basic configuration.
    #     """
    #     # Mock MPI communication
    #     comm_mock = MagicMock()
    #     comm_mock.Get_rank.return_value = 0
    #     comm_mock.Barrier.return_value = None
    #     mock_comm.return_value = comm_mock

    #     # Set up benchmark backlog and repetitions
    #     benchmark_backlog = [{"config": {"param": "value"}, "submodule": {}}]
    #     repetitions = 2

    #     # Call run_benchmark
    #     self.benchmark_manager.run_benchmark(benchmark_backlog, repetitions)

    #     # Assertions
    #     mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    #     mock_preprocess.assert_called()
    #     mock_postprocess.assert_called()
    #     self.assertEqual(mock_open.call_count, 2)
