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
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from quark.ConfigManager import ConfigManager
from quark.BenchmarkRecord import BenchmarkRecord, BenchmarkRecordStored
from quark.Plotter import Plotter
from quark.modules.Core import Core
from quark.utils import get_git_revision
from quark.utils_mpi import get_comm

comm = get_comm()


class Instruction(Enum):
    PROCEED = 0
    INTERRUPT = 1


class JobStatus(Enum):
    UNDEF = 0
    INTERRUPTED = 1
    FINISHED = 2
    FAILED = 3


def _prepend_instruction(result: tuple) -> tuple[Instruction, tuple]:
    """
    If the given list does not contain an Instruction as first entry a
    PROCEED is inserted at position 0 such that it is guaranteed that
    the first entry of the returned list is an Instruction with PROCEED
    as default.

    :param result: The tuple to which the Instruction is to be prepended
    :return: The tuple with an Instruction as first entry
    """
    if isinstance(result[0], Instruction):
        return result
    else:
        return Instruction.PROCEED, *result


def postprocess(module_instance: Core, *args, **kwargs) -> tuple[Instruction, tuple]:
    """
    Wraps module_instance.postprocess such that the first entry of the
    result list is guaranteed to be an Instruction. See _prepend_instruction.

    :param module_instance: The QUARK module on which to call postprocess
    :return: The result list of module_instance.postprocess with an Instruction as first entry.
    """
    result = module_instance.postprocess(*args, **kwargs)
    return _prepend_instruction(result)


def preprocess(module_instance: Core, *args, **kwargs) -> tuple[Instruction, tuple]:
    """
    Wraps module_instance.preprocess such that the first entry of the
    result list is guaranteed to be an Instruction. See _prepend_instruction.

    :param module_instance: The QUARK module on which to call preprocess
    :return: The result list of module_instance.preprocess with an Instruction as first entry.
    """
    result = module_instance.preprocess(*args, **kwargs)
    return _prepend_instruction(result)


class BenchmarkManager:
    """
    The benchmark manager is the main component of QUARK, orchestrating the overall benchmarking process.
    Based on the configuration, the benchmark manager will create an experimental plan considering all combinations of
    configurations, e.g., different problem sizes, solver, and hardware combinations. It will then instantiate the
    respective framework components. After executing the benchmarks, it collects the generated data and saves it.
    """

    def __init__(self, fail_fast: bool = False):
        """
        Constructor method.

        :param fail_fast: Boolean whether a single failed benchmark run causes QUARK to fail
        """
        self.fail_fast = fail_fast
        self.application = None
        self.application_configs = None  # TODO Seems to be unused, maybe delete
        self.results = []
        self.store_dir = None
        self.benchmark_record_template = None
        self.interrupted_results_path = None

    def load_interrupted_results(self) -> Optional[list]:
        """
        Loads the interrupted results if available.

        :return: The content of the results file from the QUARK run to be resumed or None.
        """
        if self.interrupted_results_path is None or not os.path.exists(self.interrupted_results_path):
            return None
        with open(self.interrupted_results_path, encoding='utf-8') as results_file:
            results = json.load(results_file)
        return results

    def _create_store_dir(self, store_dir: str = None, tag: str = None) -> None:
        """
        Creates directory for a benchmark run.

        :param store_dir: Directory where the new directory should be created
        :param tag: Prefix of the new directory
        """
        if store_dir is None:
            store_dir = Path.cwd()
        self.store_dir = f"{store_dir}/benchmark_runs/{tag + '-' if not None else ''}" \
            f"{datetime.today().strftime('%Y-%m-%d-%H-%M-%S')}"
        Path(self.store_dir).mkdir(parents=True, exist_ok=True)
        self._set_logger()

    def _resume_store_dir(self, store_dir: str) -> None:
        """
        Resumes the existing store directory.

        :param store-dir: Directory to be resumed
        """
        self.store_dir = store_dir
        self._set_logger()

    def _set_logger(self) -> None:
        """
        Sets up the logger to also write to a file in the store directory.
        """
        logger = logging.getLogger()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        filehandler = logging.FileHandler(f"{self.store_dir}/logging.log")
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)

    def orchestrate_benchmark(self, benchmark_config_manager: ConfigManager, app_modules: list[dict],
                              store_dir: str = None, interrupted_results_path: str = None) -> None:
        """
        Executes the benchmarks according to the given settings.

        :param benchmark_config_manager: Instance of BenchmarkConfigManager class, where config is already set
        :param app_modules: The list of application modules as specified in the application modules configuration
        :param store_dir: Target directory to store the results of the benchmark (if user decided to store it)
        :param interrupted_results_path: Result file from which the information for the interrupted jobs will be read.
                                         If store_dir is None the parent directory of interrupted_results_path will
                                         be used as store_dir.
        """
        self.interrupted_results_path = interrupted_results_path
        if interrupted_results_path and not store_dir:
            self._resume_store_dir(os.path.dirname(interrupted_results_path))
        else:
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

    def run_benchmark(self, benchmark_backlog: list, repetitions: int) -> None:  # pylint: disable=R0915
        """
        Goes through the benchmark backlog, which contains all the benchmarks to execute.

        :param repetitions: Number of repetitions
        :param benchmark_backlog: List with the benchmark items to run
        """
        git_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", )
        git_revision_number, git_uncommitted_changes = get_git_revision(git_dir)
        break_flag = False

        job_status_count_total = {}
        interrupted_results = self.load_interrupted_results()
        for idx_backlog, backlog_item in enumerate(benchmark_backlog):
            benchmark_records: list[BenchmarkRecord] = []
            path = f"{self.store_dir}/benchmark_{idx_backlog}"
            Path(path).mkdir(parents=True, exist_ok=True)
            with open(f"{path}/application_config.json", 'w') as filehandler:
                json.dump(backlog_item["config"], filehandler, indent=2)
            job_status_count = {}

            for i in range(1, repetitions + 1):
                logging.info(f"Running backlog item {idx_backlog + 1}/{len(benchmark_backlog)},"
                             f" Iteration {i}/{repetitions}:")

                # Getting information of interrupted jobs
                job_info_with_meta_data = {}
                if interrupted_results:
                    for entry in interrupted_results:
                        if entry["benchmark_backlog_item_number"] == idx_backlog and entry["repetition"] == i:
                            job_info_with_meta_data = entry
                            break
                job_info = job_info_with_meta_data['module'] if job_info_with_meta_data else {}
                quark_job_status_name = job_info.get("quark_job_status")

                if quark_job_status_name in (JobStatus.FINISHED.name, JobStatus.FAILED.name):
                    quark_job_status = JobStatus.FINISHED if quark_job_status_name == JobStatus.FINISHED.name \
                        else JobStatus.FAILED
                    benchmark_records.append(BenchmarkRecordStored(job_info_with_meta_data))
                    job_status_count[quark_job_status] = job_status_count.get(quark_job_status, 0) + 1
                    job_status_count_total[quark_job_status] = job_status_count_total.get(quark_job_status, 0) + 1
                    logging.info("job already %s - skip.", quark_job_status_name)
                    continue

                try:
                    self.benchmark_record_template = BenchmarkRecord(
                        idx_backlog,
                        datetime.today().strftime('%Y-%m-%d-%H-%M-%S'),
                        git_revision_number, git_uncommitted_changes,
                        i, repetitions
                    )
                    self.application.metrics.set_module_config(backlog_item["config"])
                    instruction, problem, preprocessing_time = preprocess(
                        self.application, None, backlog_item["config"],
                        store_dir=path, rep_count=i, previous_job_info=job_info
                    )
                    self.application.metrics.set_preprocessing_time(preprocessing_time)
                    self.application.save(path, i)

                    postprocessing_time = 0.0
                    benchmark_record = self.benchmark_record_template.copy()
                    if instruction == Instruction.PROCEED:
                        instruction, processed_input, benchmark_record = \
                            self.traverse_config(backlog_item["submodule"], problem,
                                                 path, rep_count=i, previous_job_info=job_info)
                        if instruction == Instruction.PROCEED:
                            instruction, _, postprocessing_time = \
                                postprocess(self.application, processed_input, backlog_item["config"],
                                            store_dir=path, rep_count=i, previous_job_info=job_info)

                    if instruction == Instruction.INTERRUPT:
                        quark_job_status = JobStatus.INTERRUPTED
                    else:
                        quark_job_status = JobStatus.FINISHED

                    self.application.metrics.add_metric("quark_job_status", quark_job_status.name)
                    self.application.metrics.set_postprocessing_time(postprocessing_time)
                    self.application.metrics.validate()

                    if benchmark_record is not None:
                        benchmark_record.append_module_record_left(deepcopy(self.application.metrics))
                        benchmark_records.append(benchmark_record)

                except KeyboardInterrupt:
                    logging.warning("CTRL-C detected during run_benchmark. Still trying to create results.json.")
                    break_flag = True
                    break

                except Exception as error:
                    logging.exception(f"Error during benchmark run: {error}", exc_info=True)
                    quark_job_status = JobStatus.FAILED
                    if job_info:
                        # Restore results/infos from previous run
                        benchmark_records.append(job_info)
                    if self.fail_fast:
                        raise

                job_status_count[quark_job_status] = job_status_count.get(quark_job_status, 0) + 1
                job_status_count_total[quark_job_status] = job_status_count_total.get(quark_job_status, 0) + 1

            for record in benchmark_records:
                record.sum_up_times()

            for record in benchmark_records:
                record.sum_up_times()

            status_report = " ".join([f"{status.name}:{count}" for status, count in job_status_count.items()])
            logging.info("")
            logging.info(f" ==== Run backlog item {idx_backlog + 1}/{len(benchmark_backlog)} "
                         f"with {repetitions} iterations - {status_report} ==== ")
            logging.info("")

            # Wait until all MPI processes have finished and save results on rank 0
            comm.Barrier()
            if comm.Get_rank() == 0:
                with open(f"{path}/results.json", 'w') as filehandler:
                    json.dump([x.get() for x in benchmark_records], filehandler, indent=2, cls=NumpyEncoder)

            logging.info("")
            logging.info(" =============== Run finished =============== ")
            logging.info("")

            if break_flag:
                break

        # Log overall status information
        status_report = " ".join([f"{status.name}:{count}" for status, count in job_status_count_total.items()])
        logging.info(80 * "=")
        logging.info(f"====== Run {len(benchmark_backlog)} backlog items "
                     f"with {repetitions} iterations - {status_report}")

        nb_interrupted = job_status_count_total.get(JobStatus.INTERRUPTED, 0)
        nb_not_started = sum(job_status_count_total.values()) < len(benchmark_backlog)
        if nb_interrupted + nb_not_started > 0:
            try:
                rel_path = Path(self.store_dir).relative_to(os.getcwd())
            except ValueError:
                rel_path = self.store_dir
            logging.info("====== There are interrupted jobs. You may resume them by running QUARK with")
            logging.info(f"====== --resume-dir={rel_path}")
        logging.info(80 * "=")
        logging.info("")

    # pylint: disable=R0917
    def traverse_config(self, module: dict, input_data: any, path: str, rep_count: int, previous_job_info:
                        dict = None) -> tuple[Instruction, any, BenchmarkRecord]:
        """
        Executes a benchmark by traversing down the initialized config recursively until it reaches the end. Then
        traverses up again. Once it reaches the root/application, a benchmark run is finished.

        :param module: Current module
        :param input_data: The input data needed to execute the current module
        :param path: Path in case the modules want to store anything
        :param rep_count: The iteration count
        :param previous_job_info: Information about previous job
        :return: Tuple with the output of this step and the according BenchmarkRecord
        """
        # Only the value of the dict is needed (dict has only one key)
        module = module[next(iter(module))]
        module_instance: Core = module["instance"]

        submodule_job_info = None
        if previous_job_info and previous_job_info.get("submodule"):
            assert module['name'] == previous_job_info["submodule"]["module_name"], \
                f"asyncronous job info given, but no information about module {module['name']} stored in it"  # TODO
            if 'submodule' in previous_job_info and previous_job_info['submodule']:
                submodule_job_info = previous_job_info['submodule']

        module_instance.metrics.set_module_config(module["config"])
        instruction, module_instance.preprocessed_input, preprocessing_time = preprocess(
            module_instance, input_data,
            module["config"], store_dir=path,
            rep_count=rep_count,
            previous_job_info=submodule_job_info
        )

        module_instance.metrics.set_preprocessing_time(preprocessing_time)
        output = None
        benchmark_record = self.benchmark_record_template.copy()
        postprocessing_time = 0.0

        if instruction == Instruction.PROCEED:
            # Check if end of the chain is reached
            if not module["submodule"]:
                # If we reach the end of the chain we create the benchmark record, fill it and then pass it up
                instruction, module_instance.postprocessed_input, postprocessing_time = postprocess(
                    module_instance,
                    module_instance.preprocessed_input,
                    module["config"], store_dir=path,
                    rep_count=rep_count,
                    previous_job_info=submodule_job_info
                )
                output = module_instance.postprocessed_input
            else:
                instruction, processed_input, benchmark_record = self.traverse_config(
                    module["submodule"],
                    module_instance.preprocessed_input, path,
                    rep_count, previous_job_info=submodule_job_info
                )

                if instruction == Instruction.PROCEED:
                    instruction, module_instance.postprocessed_input, postprocessing_time = postprocess(
                        module_instance, processed_input,
                        module["config"], store_dir=path,
                        rep_count=rep_count,
                        previous_job_info=submodule_job_info
                    )
                    output = module_instance.postprocessed_input
                else:
                    output = processed_input

        module_instance.metrics.set_postprocessing_time(postprocessing_time)
        module_instance.metrics.validate()
        benchmark_record.append_module_record_left(deepcopy(module_instance.metrics))

        return instruction, output, benchmark_record

    def _collect_all_results(self) -> list[dict]:
        """
        Collect all results from the multiple results.json.

        :return: List of dicts with results
        """
        results = []
        for filename in glob.glob(f"{self.store_dir}/**/results.json"):
            with open(filename) as f:
                results += json.load(f)

        if len(results) == 0:
            logging.error("No results.json files were found! Probably an error occurred during execution.")
        return results

    def _save_as_json(self, results: list) -> None:
        """
        Saves benchmark results to a JSON file.

        :param results: Benchmark results to be saved
        """
        logging.info(f"Saving {len(results)} benchmark records to {self.store_dir}/results.json")
        with open(f"{self.store_dir}/results.json", 'w') as filehandler:
            json.dump(results, filehandler, indent=2)

    def summarize_results(self, input_dirs: list) -> None:
        """
        Helper function to summarize multiple experiments.

        :param input_dirs: List of directories
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
        :return: A list
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
    Encoder that is used for json.dump(...) since numpy value items in dictionary might cause problems.
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
