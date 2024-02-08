#  Copyright 2021 The QUARK Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import glob
import json
import logging
import os
import os.path
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np

from ConfigManager import ConfigManager
from BenchmarkRecord import BenchmarkRecord
from Plotter import Plotter
from modules.Core import Core
from utils import get_git_revision

from utils_mpi import get_comm

comm = get_comm()



class BenchmarkManager:
    """
    The benchmark manager is the main component of QUARK, orchestrating the overall benchmarking process.
    Based on the configuration, the benchmark manager will create an experimental plan considering all combinations of
    configurations, e.g., different problem sizes, solver, and hardware combinations. It will then instantiate the
    respective framework components. After executing the benchmarks, it collects the generated data and saves it.
    """

    def __init__(self, fail_fast: bool = False):
        """
        Constructor method
        :param fail_fast: Boolean whether a single failed benchmark run causes QUARK to fail
        :type fail_fast: bool
        """
        self.fail_fast = fail_fast
        self.application = None
        self.application_configs = None
        self.results = []
        self.store_dir = None
        self.benchmark_record_template = None

    def _create_store_dir(self, store_dir: str = None, tag: str = None) -> None:
        """
        Creates directory for a benchmark run.

        :param store_dir: Directory where the new directory should be created
        :type store_dir: str
        :param tag: prefix of the new directory
        :type tag: str
        :return:
        :rtype: None
        """
        if store_dir is None:
            store_dir = Path.cwd()
        self.store_dir = f"{store_dir}/benchmark_runs/{tag + '-' if not None else ''}" \
                         f"{datetime.today().strftime('%Y-%m-%d-%H-%M-%S')}"
        Path(self.store_dir).mkdir(parents=True, exist_ok=True)

        # Also store the log file to the benchmark dir
        logger = logging.getLogger()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        filehandler = logging.FileHandler(f"{self.store_dir}/logging.log")
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)

    def orchestrate_benchmark(self, benchmark_config_manager: ConfigManager, app_modules: list[dict],
                              store_dir: str = None) -> None:
        """
        Executes the benchmarks according to the given settings.

        :param benchmark_config_manager: Instance of BenchmarkConfigManager class, where config is already set.
        :type benchmark_config_manager: ConfigManager
        :param app_modules: the list of application modules as specified in the application modules configuration.
        :type app_modules: list of dict
        :param store_dir: target directory to store the results of the benchmark (if you decided to store it)
        :type store_dir: str
        :rtype: None
        """

        self._create_store_dir(store_dir, tag=benchmark_config_manager.get_config()["application"]["name"].lower())
        benchmark_config_manager.save(self.store_dir)
        benchmark_config_manager.load_config(app_modules)
        self.application = benchmark_config_manager.get_app()
        benchmark_config_manager.create_tree_figure(self.store_dir)

        logging.info(f"Created Benchmark run directory {self.store_dir}")

        benchmark_backlog = benchmark_config_manager.start_create_benchmark_backlog()

        self.run_benchmark(benchmark_backlog, benchmark_config_manager.get_reps())

        # Wait until all MPI processes have finished and save results on rank 0
        comm.Barrier()
        if comm.Get_rank() == 0:
            results = self._collect_all_results()
            self._save_as_json(results)

    def run_benchmark(self, benchmark_backlog: list, repetitions: int):
        """
        Goes through the benchmark backlog, which contains all the benchmarks to execute.

        :param repetitions: Number of repetitions
        :type repetitions: int
        :param benchmark_backlog: List with the benchmark items to run
        :type benchmark_backlog: list
        :return:
        """
        git_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", )
        git_revision_number, git_uncommitted_changes = get_git_revision(git_dir)

        try:
            for idx_backlog, backlog_item in enumerate(benchmark_backlog):
                benchmark_records: [BenchmarkRecord] = []
                path = f"{self.store_dir}/benchmark_{idx_backlog}"
                Path(path).mkdir(parents=True, exist_ok=True)
                with open(f"{path}/application_config.json", 'w') as filehandler:
                    json.dump(backlog_item["config"], filehandler, indent=2)
                for i in range(1, repetitions + 1):
                    logging.info(f"Running backlog item {idx_backlog + 1}/{len(benchmark_backlog)},"
                                 f" Iteration {i}/{repetitions}:")
                    try:

                        self.benchmark_record_template = BenchmarkRecord(idx_backlog,
                                                                         datetime.today().strftime('%Y-%m-%d-%H-%M-%S'),
                                                                         git_revision_number, git_uncommitted_changes,
                                                                         i, repetitions)
                        self.application.metrics.set_module_config(backlog_item["config"])
                        problem, preprocessing_time = self.application.preprocess(None, backlog_item["config"],
                                                                                  store_dir=path, rep_count=i)
                        self.application.metrics.set_preprocessing_time(preprocessing_time)
                        self.application.save(path, i)

                        processed_input, benchmark_record = self.traverse_config(backlog_item["submodule"], problem,
                                                                                 path, rep_count=i)

                        _, postprocessing_time = self.application.postprocess(processed_input, None, store_dir=path,
                                                                              rep_count=i)
                        self.application.metrics.set_postprocessing_time(postprocessing_time)
                        self.application.metrics.validate()
                        benchmark_record.append_module_record_left(deepcopy(self.application.metrics))
                        benchmark_records.append(benchmark_record)

                    except Exception as error:
                        logging.exception(f"Error during benchmark run: {error}", exc_info=True)
                        if self.fail_fast:
                            raise

                for record in benchmark_records:
                    record.sum_up_times()

                # Wait until all MPI processes have finished and save results on rank 0
                comm.Barrier()
                if comm.Get_rank() == 0:
                    with open(f"{path}/results.json", 'w') as filehandler:
                        json.dump([x.get() for x in benchmark_records], filehandler, indent=2, cls=NumpyEncoder)

                logging.info("")
                logging.info(" =============== Run finished =============== ")
                logging.info("")

        except KeyboardInterrupt:
            logging.warning("CTRL-C detected. Still trying to create results.json.")

    def traverse_config(self, module: dict, input_data: any, path: str, rep_count: int) -> (any, BenchmarkRecord):
        """
        Executes a benchmark by traversing down the initialized config recursively until it reaches the end. Then
        traverses up again. Once it reaches the root/application, a benchmark run is finished.

        :param module: Current module
        :type module: dict
        :param input_data: The input data needed to execute the current module.
        :type input_data: any
        :param path: Path in case the modules want to store anything
        :type path: str
        :param rep_count: The iteration count
        :type rep_count: int
        :return: tuple with the output of this step and the according BenchmarkRecord
        :rtype: tuple(any, BenchmarkRecord)
        """

        # Only the value of the dict is needed (dict has only one key)
        module = module[next(iter(module))]
        module_instance: Core = module["instance"]

        module_instance.metrics.set_module_config(module["config"])
        module_instance.preprocessed_input, preprocessing_time = module_instance.preprocess(input_data,
                                                                                            module["config"],
                                                                                            store_dir=path,
                                                                                            rep_count=rep_count)
        module_instance.metrics.set_preprocessing_time(preprocessing_time)

        # Check if end of the chain is reached
        if not module["submodule"]:
            # If we reach the end of the chain we create the benchmark record, fill it and then pass it up
            benchmark_record = self.benchmark_record_template.copy()
            module_instance.postprocessed_input, postprocessing_time = module_instance.postprocess(
                module_instance.preprocessed_input, module["config"], store_dir=path, rep_count=rep_count)

        else:
            processed_input, benchmark_record = self.traverse_config(module["submodule"],
                                                                     module_instance.preprocessed_input, path,
                                                                     rep_count)
            module_instance.postprocessed_input, postprocessing_time = module_instance.postprocess(processed_input,
                                                                                                   module["config"],
                                                                                                   store_dir=path,
                                                                                                   rep_count=rep_count)

        output = module_instance.postprocessed_input
        module_instance.metrics.set_postprocessing_time(postprocessing_time)
        module_instance.metrics.validate()
        benchmark_record.append_module_record_left(deepcopy(module_instance.metrics))

        return output, benchmark_record

    def _collect_all_results(self) -> List[Dict]:
        """
        Collect all results from the multiple results.json.

        :return: list of dicts with results
        :rtype: List[Dict]
        """
        results = []
        for filename in glob.glob(f"{self.store_dir}/**/results.json"):
            with open(filename) as f:
                results += json.load(f)

        if len(results) == 0:
            logging.error("No results.json files were found! Probably an error occurred during execution.")
        return results

    def _save_as_json(self, results: list) -> None:
        logging.info(f"Saving {len(results)} benchmark records to {self.store_dir}/results.json")
        with open(f"{self.store_dir}/results.json", 'w') as filehandler:
            json.dump(results, filehandler, indent=2)

    def summarize_results(self, input_dirs: list) -> None:
        """
        Helper function to summarize multiple experiments.

        :param input_dirs: list of directories
        :type input_dirs: list
        :rtype: None
        """
        self._create_store_dir(tag="summary")
        logging.info(f"Summarizing {len(input_dirs)} benchmark directories")
        results = self.load_results(input_dirs)
        self._save_as_json(results)
        Plotter.visualize_results(results, self.store_dir)

    def load_results(self, input_dirs: list = None) -> list:
        """
        Load results from one or more results.json files.

        :param input_dirs: If you want to load more than 1 results.json (default is just 1, the one from the experiment)
        :type input_dirs: list
        :return: a list
        :rtype: list
        """

        if input_dirs is None:
            input_dirs = [self.store_dir]

        results = []
        for input_dir in input_dirs:
            for filename in glob.glob(f"{input_dir}/results.json"):
                with open(filename) as f:
                    results += json.load(f)

        return results


class NumpyEncoder(json.JSONEncoder):
    """
    Encoder that is used for json.dump(...) since numpy value items in dictionary might cause problems
    """

    def default(self, o: any):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)
