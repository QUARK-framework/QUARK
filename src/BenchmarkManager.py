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
import itertools
import importlib
import json
import logging
import os
import os.path
import re
import sys
from datetime import datetime
from pathlib import Path
import inquirer
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
import pandas as pd
import seaborn as sns
import yaml
import subprocess

matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('font', family='serif')
matplotlib.rcParams['savefig.dpi'] = 300
sns.set_style('darkgrid')
sns.color_palette()


def _import_class(module_path: str, class_name: str, base_dir: str = None) -> type:
    """
    Helper function which allows to replace hard coded imports of the form
    'import MyClass from path.to.mypkg' by calling _import_class('path.to.mypkg', 'MyClass').
    If base_dir is specified, its value will be added to the python search path
    if not already contained in it.

    :param module_path: python module path of the module containing the class to be imported
    :type module_path: str
    :param class_name: the name of the class to be imported
    :type class_name: str
    :return: the imported class object
    :rtype: type
    """

    # make sure that base_dir is in the search path. Otherwise, the module imported here might not find its libraries.
    if base_dir is not None and base_dir not in sys.path:
        logging.info(f"Appending to sys.path: {base_dir}")
        sys.path.append(base_dir)
    logging.info(f"Importing module {module_path}")
    module = importlib.import_module(module_path)
    return vars(module)[class_name]


def _get_instance_with_sub_options(options: list, class_name: str, *args: any) -> any:
    """
    Create an instance of the QUARK module (application, mapping, solver, device) identified by
    class_name.

    :param options: the section of the QUARK module configuration which is relevant here, including the information on submodules.
    :type options: list of dict
    :param class_name: the name of the class to be initialized.
    :type class_name: str
    :param args: the list of arguments used for to the class initialization
    :type args: any
    :return: the new instance of the QUARK module
    :rtype: any
    """
    for opt in options:
        if "class" in opt:
            class_name = opt["class"]
        elif class_name != opt["name"]:
            continue
        clazz = _import_class(opt["module"], class_name, opt.get("dir"))
        sub_options = None
        for key in ["mappings", "solvers", "devices"]:
            if key in opt:
                sub_options = opt[key]
                break
        # In case the class requires some arguments in its constructor they can be defined in the "args" dict
        if "args" in opt:
            instance = clazz(**opt["args"])
        else:
            instance = clazz(*args)

        # sub_options inherits 'dir'
        if sub_options is not None and "dir" in opt:
            for sub_opt in sub_options:
                if "dir" not in sub_opt:
                    sub_opt["dir"] = opt["dir"]

        instance.sub_options = sub_options
        return instance
    logging.warning(f"{class_name} not found in {options}")


class BenchmarkManager:
    """
    The benchmark manager is the main component of QUARK orchestrating the overall benchmarking process.
    Based on the configuration, the benchmark manager will create an experimental plan considering all combinations of
    configurations, e.g., different problem sizes, solver, and hardware combinations. It will then instantiate the
    respective framework components representing the application, the mapping to the algorithmic formulation, solver,
    and device. After executing the benchmarks, it collects the generated data and executes the validation and evaluation
    functions.
    """

    def __init__(self):
        """
        Constructor method
        """
        self.application = None
        self.application_configs = None
        self.results = []
        self.mapping_solver_device_combinations = {}
        self.repetitions = 1
        self.store_dir = None

    def generate_benchmark_configs(self, app_modules: list) -> dict:
        """
        Queries the user to get all needed information about application, solver, mapping, device and general settings
        to run the benchmark.

        :param app_modules: the list of application modules as specified in the application modules configuration.
        :type app_modules: list of dict
        :return: Benchmark Config
        :rtype: dict
        """
        application_answer = inquirer.prompt([inquirer.List('application',
                                                            message="What application do you want?",
                                                            choices=[m["name"] for m in app_modules],
                                                            default='PVC',
                                                            )])

        app_name = application_answer["application"]
        self.application = _get_instance_with_sub_options(app_modules, app_name)

        application_config = self.application.get_parameter_options()

        application_config = BenchmarkManager._query_for_config(application_config,
                                                                f"(Option for {application_answer['application']})")

        config = {
            "application": {
                "name": application_answer["application"],
                "config": application_config
            },
            "mapping": {}
        }

        mapping_answer = BenchmarkManager.checkbox(key='mapping',
                                                   message="What mapping do you want?",
                                                   choices=self.application.get_available_mapping_options())

        for mapping_single_answer in mapping_answer["mapping"]:
            mapping = self.application.get_submodule(mapping_single_answer)

            mapping_config = mapping.get_parameter_options()
            mapping_config = BenchmarkManager._query_for_config(mapping_config, f"(Option for {mapping_single_answer})")

            solver_answer = BenchmarkManager.checkbox(key='solver',
                                                      message=f"What Solver do you want for mapping {mapping_single_answer}?",
                                                      choices=mapping.get_available_solver_options())
            config["mapping"][mapping_single_answer] = {
                "solver": [],
                "config": mapping_config
            }

            for solver_single_answer in solver_answer["solver"]:
                solver = mapping.get_submodule(solver_single_answer)
                solver_config = solver.get_parameter_options()
                solver_config = BenchmarkManager._query_for_config(solver_config,
                                                                   f"(Option for {solver_single_answer})")

                device_answer = BenchmarkManager.checkbox(key='device',
                                                          message=f"What Device do you want for solver {solver_single_answer}?",
                                                          choices=solver.get_available_device_options())

                config["mapping"][mapping_single_answer]["solver"].append({
                    "name": solver_single_answer,
                    "config": solver_config,
                    "device": device_answer["device"]

                })

        repetitions_answer = inquirer.prompt(
            [inquirer.Text('repetitions', message="How many repetitions do you want?",
                           validate=lambda _, x: re.match("\\d", x),
                           default=self.repetitions
                           )])

        config['repetitions'] = int(repetitions_answer["repetitions"])

        logging.info(config)
        return config

    def load_config(self, config: dict, app_modules: list) -> None:
        """
        Uses the config file to generate all class instances needed to run the benchmark.

        :param config: valid config file
        :type config: dict
        :param app_modules: the list of application modules as specified in the application modules configuration.
        :type app_modules: list of dict
        :rtype: None
        """

        logging.info(config)

        app_name = config["application"]["name"]
        self.application = _get_instance_with_sub_options(app_modules, app_name)

        self.repetitions = int(config["repetitions"])

        # Build all application configs
        keys, values = zip(*config['application']['config'].items())
        self.application_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
        self.mapping_solver_device_combinations = {}

        for mapping_name, mapping_value in config['mapping'].items():
            mapping = self.application.get_submodule(mapping_name)

            if len(mapping_value['config'].items()) > 0:
                keys, values = zip(*mapping_value['config'].items())
                mapping_config = [dict(zip(keys, v)) for v in itertools.product(*values)]
            else:
                mapping_config = [{}]

            self.mapping_solver_device_combinations[mapping_name] = {
                "mapping_instance": mapping,
                "mapping_config": mapping_config,
                "solvers": {}
            }
            for single_solver in mapping_value['solver']:
                # Build all solver configs
                if len(single_solver['config'].items()) > 0:
                    keys, values = zip(*single_solver['config'].items())
                    solver_config = [dict(zip(keys, v)) for v in itertools.product(*values)]
                else:
                    solver_config = [{}]
                solver = mapping.get_submodule(single_solver['name'])
                self.mapping_solver_device_combinations[mapping_name]["solvers"][single_solver['name']] = {
                    "solver_instance": solver,
                    "solver_config": solver_config
                }

                self.mapping_solver_device_combinations[mapping_name]["solvers"][single_solver['name']][
                    "devices"] = {}

                for single_device in single_solver["device"]:
                    device_wrapper = solver.get_submodule(single_device)
                    self.mapping_solver_device_combinations[mapping_name]["solvers"][single_solver['name']][
                        "devices"][single_device] = device_wrapper

    @staticmethod
    def _query_for_config(param_opts: dict, prefix: str = "") -> dict:
        config = {}
        for key, config_answer in param_opts.items():
            if config_answer.get("if"):
                # support parameter descriptions like
                # "seed": {
                #     "if": {"key":"graph_type", "in" : ["erdos-renyi"]},
                #     ...
                # }
                # meaning that 'seed' only gets displayed if graph_type has been chosen to be 'erdos-renyi'
                # This expects that the referenced parameter has been declared before and is declared to be
                # 'exclusive' so that its value is unique.

                key_in_cond = config_answer.get("if")["key"]
                dependency = param_opts.get(key_in_cond)

                # check if configuration is consistent
                consistent = False
                err_msg = None
                if dependency is None:
                    err_msg = f"Inconsistent parameter options: condition references unknown parameter: {key_in_cond}"
                elif not dependency.get('exclusive', False):
                    err_msg = f"Inconsistent parameter options: condition references non exclusive parameter: {key_in_cond}"
                else:
                    consistent = True
                if not consistent:
                    raise Exception(f"{prefix} {err_msg}")

                if not config[key_in_cond][0] in config_answer.get("if")["in"]:
                    continue

            if len(config_answer['values']) == 1:
                # When there is only 1 value to choose from skip the user input for now
                values = config_answer['values']
                print(f"{prefix} {config_answer['description']}: {config_answer['values'][0]}")

            elif config_answer.get('exclusive', False):
                answer = inquirer.prompt(
                    [inquirer.List(key,
                                   message=f"{prefix} {config_answer['description']}",
                                   choices=config_answer['values']
                                   )])
                values = (answer[key],)
            else:
                answer = BenchmarkManager.checkbox(key=key,
                                                   message=f"{prefix} {config_answer['description']}",
                                                   choices=config_answer['values'])
                values = answer[key]

            if config_answer.get("postproc"):
                # the value of config_answer.get("postproc") is expected to be callable
                # with each of the user selected values as argument.
                # Note that the stored config file will contain the processed values.
                values = [config_answer["postproc"](v) for v in values]
            config[key] = values
        return config

    @staticmethod
    def checkbox(key: str, message: str, choices: list) -> dict:
        """
        Wrapper method to avoid empty responses in checkbox.

        :param key: Key for response dict
        :type key: str
        :param message: Message for the user
        :type message: str
        :param choices: Choices for the user
        :type choices: list
        :return: Dict with the response from the user
        :rtype: dict
        """

        answer = inquirer.prompt([inquirer.Checkbox(key, message=message, choices=choices)])

        if not answer[key]:
            logging.warning("You need to check at least one box! Please try again!")
            return BenchmarkManager.checkbox(key, message, choices)
        else:
            return answer

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
        self.store_dir = f"{store_dir}/benchmark_runs/{tag + '-' if not None else ''}{datetime.today().strftime('%Y-%m-%d-%H-%M-%S')}"
        Path(self.store_dir).mkdir(parents=True, exist_ok=True)

    def orchestrate_benchmark(self, config: dict, app_modules: list, store_dir: str = None) -> None:
        """
        Executes the benchmarks according to the given settings.

        :param config: valid config file
        :type config: dict
        :param app_modules: the list of application modules as specified in the application modules configuration.
        :type app_modules: list of dict
        :param store_dir: target directory to store the results of the benchmark (if you decided to store it)
        :type store_dir: str
        :rtype: None
        """
        # TODO Make this nicer

        self.load_config(config, app_modules)

        self._create_store_dir(store_dir, tag=self.application.__class__.__name__.lower())
        logger = logging.getLogger()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh = logging.FileHandler(f"{self.store_dir}/logger.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logging.info(f"Created Benchmark run directory {self.store_dir}")

        # Collect git revision number and check if there are uncommitted changes to allow user to analyze which
        # codebase was used for benchmark runs
        try:
            # TODO: Does it work with windows?
            # '-C', git_dir ensures that the following commands also work when QUARK is started from other working
            # directories
            git_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", )
            git_revision_number = subprocess.check_output(['git', '-C', git_dir, 'rev-parse', 'HEAD']).decode(
                'ascii').strip()
            git_uncommitted_changes = True if subprocess.check_output(
                ['git', '-C', git_dir, 'status', '--porcelain', '--untracked-files=no']).decode(
                'ascii').strip() else False

            logging.info(
                f"Codebase is based on revision {git_revision_number} and has {'some' if git_uncommitted_changes else 'no'} uncommitted changes")
        except Exception as e:
            logging.warning(f"Logging of git revision number not possible because of: {e}")
            git_revision_number = "unknown"
            git_uncommitted_changes = "unknown"

        with open(f"{self.store_dir}/config.yml", 'w') as fp:
            yaml.dump(config, fp)
        try:
            for idx, application_config in enumerate(self.application_configs):
                results = []

                path = f"{self.store_dir}/application_config_{idx}"
                Path(path).mkdir(parents=True, exist_ok=True)
                with open(f"{path}/application_config.json", 'w') as fp:
                    json.dump(application_config, fp)
                for mapping_name, mapping_value in self.mapping_solver_device_combinations.items():
                    mapping = mapping_value["mapping_instance"]
                    for mapping_config in mapping_value['mapping_config']:
                        for solver_name, solver_value in mapping_value["solvers"].items():
                            solver = solver_value["solver_instance"]
                            for solver_config in solver_value['solver_config']:
                                for device_name, device_value in solver_value["devices"].items():
                                    device = device_value
                                    for i in range(1, self.repetitions + 1):
                                        problem = self.application.init_problem(application_config, idx, i, path)
                                        mapped_problem, time_to_mapping = mapping.map(problem, mapping_config)
                                        try:
                                            logging.info(
                                                f"Running {self.application.__class__.__name__} with config "
                                                f"{application_config} on solver {solver.__class__.__name__} and device "
                                                f"{device.get_device_name()} (Repetition {i}/{self.repetitions})")
                                            solution_raw, time_to_solve, additional_solver_information = solver.run(
                                                mapped_problem, device, solver_config, store_dir=path, repetition=i)
                                            processed_solution, time_to_reverse_map = mapping.reverse_map(solution_raw)
                                            try:
                                                processed_solution, time_to_process_solution = self.application.process_solution(
                                                    processed_solution)
                                                solution_validity, time_to_validation = self.application.validate(
                                                    processed_solution)
                                            except Exception as e:
                                                logging.exception("Exception on processing the solution")
                                                solution_validity = False
                                                time_to_process_solution = None
                                                time_to_validation = None
                                            if solution_validity:
                                                solution_quality, time_to_evaluation = self.application.evaluate(
                                                    processed_solution)
                                            else:
                                                solution_quality = None
                                                time_to_evaluation = None
                                            results.append({
                                                "timestamp": datetime.today().strftime('%Y-%m-%d-%H-%M-%S'),
                                                "time_to_solution": sum(filter(None, [time_to_mapping, time_to_solve,
                                                                                      time_to_reverse_map,
                                                                                      time_to_process_solution,
                                                                                      time_to_validation,
                                                                                      time_to_evaluation])),
                                                "time_to_solution_unit": "ms",
                                                "time_to_process_solution": time_to_process_solution,
                                                "time_to_process_solution_unit": "ms",
                                                "time_to_validation": time_to_validation,
                                                "time_to_validation_unit": "ms",
                                                "time_to_evaluation": time_to_evaluation,
                                                "time_to_evaluation_unit": "ms",
                                                "solution_validity": solution_validity,
                                                "solution_quality": solution_quality,
                                                "solution_quality_unit": self.application.get_solution_quality_unit(),
                                                "solution_raw": str(solution_raw),
                                                "additional_solver_information": additional_solver_information,
                                                # TODO Revise this (I am only doing this for now since json.dumps does not like tuples as keys for dicts
                                                "time_to_solve": time_to_solve,
                                                "time_to_solve_unit": "ms",
                                                "repetition": i,
                                                "application": self.application.__class__.__name__,
                                                "application_config": application_config,
                                                "mapping_config": mapping_config,
                                                "time_to_reverse_map": time_to_reverse_map,
                                                "time_to_reverse_map_unit": "ms",
                                                "time_to_mapping": time_to_mapping,
                                                "time_to_mapping_unit": "ms",
                                                "solver_config": solver_config,
                                                "mapping": mapping.__class__.__name__,
                                                "solver": solver.__class__.__name__,
                                                "device_class": device.__class__.__name__,
                                                "device": device.get_device_name(),
                                                "git_revision_number": git_revision_number,
                                                "git_uncommitted_changes ": git_uncommitted_changes
                                            })
                                            with open(f"{path}/results.json", 'w') as fp:
                                                json.dump(results, fp)
                                            df = self._collect_all_results()
                                            self._save_as_csv(df)
                                        except Exception as e:
                                            logging.error(f"Error during benchmark run: {e}", exc_info=True)
                                            with open(f"{path}/error.log", 'a') as fp:
                                                fp.write(
                                                    f"Solver: {solver_name}, Device: {device_name}, Error: {str(e)} "
                                                    f"(For more information take a look at logger.log)")
                                                fp.write("\n")

                with open(f"{path}/results.json", 'w') as fp:
                    json.dump(results, fp)
        # catching ctrl-c and killing network if desired
        except KeyboardInterrupt:
            logger.info("CTRL-C detected. Still trying to create results.csv.")
        df = self._collect_all_results()
        self._save_as_csv(df)

    def _collect_all_results(self) -> pd.DataFrame:
        """
        Collect all results from the multiple results.json.

        :return: a pandas dataframe
        :rtype: pd.Dataframe
        """
        dfs = []
        for filename in glob.glob(f"{self.store_dir}/**/results.json"):
            dfs.append(pd.read_json(filename, orient='records'))

        if len(dfs) == 0:
            logging.error("No results.json files could be found! Probably an error was previously happening.")
        return pd.concat(dfs, axis=0, ignore_index=True)

    def _save_as_csv(self, df: pd.DataFrame) -> None:
        """
        Save all the results of this experiments in a single CSV.

        :param df: Dataframe which should be saved
        :type df: pd.Dataframe
        """

        # Since these configs are dicts it is not so nice to store them in a df/csv. But this is a workaround that works for now
        df['application_config'] = df.apply(lambda row: json.dumps(row["application_config"]), axis=1)
        df['additional_solver_information'] = df.apply(lambda row: json.dumps(row["additional_solver_information"]),
                                                       axis=1)
        df['solver_config'] = df.apply(lambda row: json.dumps(row["solver_config"]), axis=1)
        df['mapping_config'] = df.apply(lambda row: json.dumps(row["mapping_config"]), axis=1)
        df.to_csv(path_or_buf=f"{self.store_dir}/results.csv")

    def load_results(self, input_dirs: list = None) -> pd.DataFrame:
        """
        Load results from one or many results.csv files.

        :param input_dirs: If you want to load more than 1 results.csv (default is just 1, the one from the experiment)
        :type input_dirs: list
        :return: a pandas dataframe
        :rtype: pd.Dataframe
        """

        if input_dirs is None:
            input_dirs = [self.store_dir]

        dfs = []
        for input_dir in input_dirs:
            for filename in glob.glob(f"{input_dir}/results.csv"):
                dfs.append(pd.read_csv(filename, index_col=0, encoding="utf-8"))

        df = pd.concat(dfs, axis=0, ignore_index=True)
        df['application_config'] = df.apply(lambda row: json.loads(row["application_config"]), axis=1)
        df['solver_config'] = df.apply(lambda row: json.loads(row["solver_config"]), axis=1)
        df['additional_solver_information'] = df.apply(lambda row: json.loads(row["additional_solver_information"]),
                                                       axis=1)
        df['mapping_config'] = df.apply(lambda row: json.loads(row["mapping_config"]), axis=1)

        return df

    def summarize_results(self, input_dirs: list) -> None:
        """
        Helper function to summarize multiple experiments.

        :param input_dirs: list of directories
        :type input_dirs: list
        :rtype: None
        """
        self._create_store_dir(tag="summary")
        df = self.load_results(input_dirs)
        # Deep copy, else it messes with the json.loads in save_as_csv
        self._save_as_csv(df.copy())
        self.visualize_results(df, self.store_dir)

    def visualize_results(self, df: pd.DataFrame, store_dir: str = None) -> None:
        """
        Generates various plots for the benchmark.

        :param df: pandas dataframe
        :type df: pd.Dataframe
        :param store_dir: directory where to store the plots
        :type store_dir: str
        :rtype: None
        """

        if store_dir is None:
            store_dir = self.store_dir

        if len(df['application'].unique()) > 1:
            logging.error("At the moment only 1 application can be visualized! Aborting plotting process!")
            return

        # Let's create some custom columns
        df['configCombo'] = df.apply(lambda row: f"{row['mapping']}/\n{row['solver']}/\n{row['device']}", axis=1)

        df, eval_axis_name = self._compute_application_config_combo(df)

        # The sorting is necessary to ensure that the solverConfigCombo is created in a consistent manner
        df['solverConfigCombo'] = df.apply(
            lambda row: '/\n'.join(
                ['%s: %s' % (key, value) for (key, value) in
                 sorted(row['solver_config'].items(), key=lambda key_value_pair: key_value_pair[0])]) +
                        "\ndevice:" + row['device'] + "\nmapping:" + '/\n'.join(
                ['%s: %s' % (key, value) for (key, value) in
                 sorted(row['mapping_config'].items(), key=lambda key_value_pair: key_value_pair[0])]), axis=1)

        df_complete = df.copy()
        df = df.loc[df["solution_validity"]]  # only keep the valid solutions

        if df.shape[0] < 1:
            logging.warning("Not enough (valid) data to visualize results, skipping the plot generation!")
            return

        self._plot_overall(df, store_dir, eval_axis_name)
        self._plot_solvers(df, store_dir, eval_axis_name)
        self._plot_solution_validity(df_complete, store_dir)

    @staticmethod
    def _compute_application_config_combo(df: pd.DataFrame) -> (pd.DataFrame, str):
        """
        Tries to infer the column and the axis name used for solution_quality in a smart way.

        :param df: pandas dataframe
        :type df: pd.Dataframe
        :return: Dataframe and the axis name
        :rtype: tuple(pd.DataFrame, str)
        """
        column = df['application_config']
        affected_keys = []
        helper_dict = defaultdict(list)
        # Try to find out which key in the dict change
        for d in column.values:  # you can list as many input dicts as you want here
            for key, value in d.items():
                helper_dict[key].append(value)
                helper_dict[key] = list(set(helper_dict[key]))

        for key, value in helper_dict.items():
            # If there is more than 1 value, and it is a float/int, then we can order it
            if len(value) > 1:  # and isinstance(value[0], (int, float))
                affected_keys.append(key)

        # def custom_sort(series):
        #     return sorted(range(len(series)), key=lambda k: tuple([series[k][x] for x in affected_keys]))
        #
        # # Sort by these keys
        # df.sort_values(by=["application_config"], key=custom_sort, inplace=True)

        if len(affected_keys) == 1:
            # X-axis name should be this and fixed parameters in parentheses
            df['applicationConfigCombo'] = df.apply(
                lambda row: row['application_config'][affected_keys[0]],
                axis=1)
            axis_name = f"{affected_keys[0]}" if len(
                helper_dict.keys()) == 1 else f"{affected_keys[0]} with {','.join(['%s %s' % (value[0], key) for (key, value) in helper_dict.items() if key not in affected_keys])}"

        else:
            # The sorting is necessary to ensure that the solverConfigCombo is created in a consistent manner
            df['applicationConfigCombo'] = df.apply(
                lambda row: '/\n'.join(['%s: %s' % (key, value) for (key, value) in
                                        sorted(row['application_config'].items(),
                                               key=lambda key_value_pair: key_value_pair[0]) if
                                        key in affected_keys]), axis=1)

            axis_name = None

        return df, axis_name

    @staticmethod
    def _plot_solution_validity(df_complete: pd.DataFrame, store_dir: str) -> None:
        """
        Generates plot for solution_validity.

        :param df_complete: pandas dataframe
        :type df_complete: pd.DataFrame
        :param store_dir: directory where to store the plot
        :type store_dir: str
        :rtype: None
        """

        def countplot(x, hue, **kwargs):
            sns.countplot(x=x, hue=hue, **kwargs)

        g = sns.FacetGrid(df_complete,
                          col="applicationConfigCombo")
        g.map(countplot, "configCombo", "solution_validity")
        g.add_legend(fontsize='7', title="Result Validity")
        g.set_ylabels("Count")
        g.set_xlabels("Solver Setting")
        for ax in g.axes.ravel():
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        g.tight_layout()

        plt.savefig(f"{store_dir}/plot_solution_validity.pdf", dpi=300)
        plt.clf()

    @staticmethod
    def _plot_solvers(df: pd.DataFrame, store_dir: str, eval_axis_name: str) -> None:
        """
        Generates plot for each individual solver.

        :param eval_axis_name: name of the evaluation metric
        :type eval_axis_name: str
        :param df: pandas dataframe
        :type df: pd.Dataframe
        :param store_dir: directory where to store the plot
        :type store_dir: str
        :rtype: None
        """

        def _barplot(data, x, y, hue=None, title="TBD", ax=None, order=None,
                     hue_order=None, capsize=None):
            sns.barplot(x=x, y=y, hue=hue, data=data, ax=ax, order=order, hue_order=hue_order,
                        capsize=capsize)  # , palette="Dark2"
            plt.title(title)
            return plt

        for solver in df['solver'].unique():

            figu, ax = plt.subplots(1, 2, figsize=(15, 10))

            _barplot(
                df.loc[df["solver"] == solver],
                "applicationConfigCombo", "time_to_solve", hue='solverConfigCombo', order=None,
                title="", ax=ax[0])
            _barplot(
                df.loc[df["solver"] == solver],
                "applicationConfigCombo", "solution_quality", hue='solverConfigCombo', order=None,
                title="", ax=ax[1])

            ax[0].get_legend().remove()
            # ax[1].get_legend().remove()
            # plt.legend(bbox_to_anchor=[1.5, .5], loc=9, frameon=False, title="Solver Settings")
            ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Solver Settings")
            ax[0].set_xlabel(xlabel=eval_axis_name, fontsize=16)
            ax[1].set_xlabel(xlabel=eval_axis_name, fontsize=16)

            ax[0].set_ylabel(ylabel=df['time_to_solve_unit'].unique()[0], fontsize=16)
            # ax[0].set_yscale('log', base=10)
            ax[1].set_ylabel(ylabel=df['solution_quality_unit'].unique()[0], fontsize=16)
            plt.suptitle(f"{solver}")

            for ax in figu.axes:
                matplotlib.pyplot.sca(ax)
                # If column values are very long and of type string rotate the ticks
                if (pd.api.types.is_string_dtype(df.applicationConfigCombo.dtype) or pd.api.types.is_object_dtype(
                        df.applicationConfigCombo.dtype)) and df.applicationConfigCombo.str.len().max() > 10:
                    plt.xticks(rotation=90)
                ax.set_xlabel(
                    xlabel=f"{df['application'].unique()[0]} Config {'(' + eval_axis_name + ')' if eval_axis_name is not None else ''}",
                    fontsize=12)

            figu.tight_layout()
            # plt.suptitle("Edge Inference: Preprocessing")
            # plt.subplots_adjust(top=0.92)
            logging.info(f"Saving plot for solver {solver}")
            plt.savefig(f"{store_dir}/plot_{solver}" + ".pdf")
            plt.clf()

    @staticmethod
    def _plot_overall(df: pd.DataFrame, store_dir: str, eval_axis_name: str) -> None:
        """
        Generates time and solution_quality plots for all solvers.

        :param eval_axis_name: name of the evaluation metric
        :type eval_axis_name: str
        :param df: pandas dataframe
        :type df: pd.Dataframe
        :param store_dir: directory where to store the plot
        :type store_dir: str
        :rtype: None
        """
        for metric in ["solution_quality", "time_to_solve"]:
            needed_col_wrap = df['solver'].nunique()

            g = sns.FacetGrid(df, col="solver", hue="solverConfigCombo", col_wrap=needed_col_wrap, legend_out=True)
            if len(df.applicationConfigCombo.unique()) < 2:
                g.map(sns.barplot, "applicationConfigCombo", metric,
                      order=df["applicationConfigCombo"])
            else:
                g.map(sns.lineplot, "applicationConfigCombo", metric, marker="X")
                g.set(xticks=list(df.applicationConfigCombo.unique()),
                      xticklabels=list(df.applicationConfigCombo.unique()))

            g.set_xlabels(
                f"{df['application'].unique()[0]} Config {'(' + eval_axis_name + ')' if eval_axis_name is not None else ''}")

            if metric == "time_to_solve":
                g.set_ylabels(df['time_to_solve_unit'].unique()[0])
                # for ax in g.axes:
                #     ax.set_yscale('log', basex=10)
            else:
                g.set_ylabels(df['solution_quality_unit'].unique()[0])
            g.add_legend(fontsize='7')

            # If column values are very long and of type string rotate the ticks
            if (pd.api.types.is_string_dtype(df.applicationConfigCombo.dtype) or pd.api.types.is_object_dtype(
                    df.applicationConfigCombo.dtype)) and df.applicationConfigCombo.str.len().max() > 10:
                for ax in g.axes.ravel():
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            g.tight_layout()

            plt.savefig(f"{store_dir}/plot_{metric}.pdf", dpi=300)
            logging.info(f"Saving plot for metric {metric}")
            plt.clf()
