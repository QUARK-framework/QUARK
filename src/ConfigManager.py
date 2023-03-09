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

import itertools
import logging
import re

import inquirer
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yaml
from typing_extensions import TypedDict, NotRequired, Self

from modules.Core import Core
from modules.applications import Application
from utils import _get_instance_with_sub_options


class ConfigModule(TypedDict):
    """
    Each Module consists of the name of the module, its config, its configured submodules, which are modules itself.
    It can also contain the instance of the class associated to this module, which is then used in the actual benchmark
    process.
    """
    name: str
    config: dict
    submodules: list[Self]
    instance: NotRequired[Core]


class BenchmarkConfig(TypedDict):
    """
    The benchmarking config consists of a Module, which is the application on the first level and the number of
    repetitions for the benchmark.
    """
    repetitions: int
    application: ConfigModule


class ConfigManager:
    """
    Classes responsible for generating/loading QUARK benchmark configs. Loading includes instantiation of the various
    modules specified in the config

    """

    def __init__(self):
        self.config = None
        self.application = None

    def generate_benchmark_configs(self, app_modules: dict):
        """
        Queries the user to get all needed information about application, solver, mapping, device and general settings
        to run the benchmark.

        :param app_modules: the list of application modules as specified in the application modules configuration.
        :type app_modules: list of dict
        """
        application_answer = inquirer.prompt([inquirer.List('application',
                                                            message="What application do you want?",
                                                            choices=[m["name"] for m in app_modules],
                                                            default='PVC',
                                                            )])

        app_name = application_answer["application"]
        self.application = _get_instance_with_sub_options(app_modules, app_name)

        application_config = self.application.get_parameter_options()

        application_config = ConfigManager._query_for_config(
            application_config, f"(Option for {application_answer['application']})")

        submodule_answer = ConfigManager.checkbox(key='submodules',
                                                  message="What submodule do you want?",
                                                  choices=self.application.get_available_submodule_options())
        self.config = {
            "application": {
                "name": app_name,
                "config": application_config,
                "submodules": [self.query_module(self.application.get_submodule(sm), sm) for sm in
                               submodule_answer["submodules"]],
            },
            "repetitions": 1  # default value

        }
        logging.info("Submodule configuration finished")

        repetitions_answer = inquirer.prompt(
            [inquirer.Text('repetitions', message="How many repetitions do you want?",
                           validate=lambda _, x: re.match("\\d", x),
                           default=1
                           )])
        self.config["repetitions"] = int(repetitions_answer["repetitions"])

    def query_module(self, module: Core, module_friendly_name: str) -> ConfigModule:
        """
        Recursive function which queries every module and its submodule until end is reached

        :param module_friendly_name: Name of the module
        :type module_friendly_name: str
        :param module: Module instance
        :type module: Core
        :return: Config module with the choices of the user
        :rtype: ConfigModule
        """

        module_config = module.get_parameter_options()
        module_config = ConfigManager._query_for_config(module_config,
                                                        f"(Option for {module.__class__.__name__})")
        available_submodules = module.get_available_submodule_options()

        if available_submodules:
            if len(available_submodules) == 1:
                logging.info(
                    f"Skipping asking for submodule, since only 1 option ({available_submodules[0]}) is available.")
                submodule_answer = {"submodules": [available_submodules[0]]}
            else:
                submodule_answer = ConfigManager.checkbox(key='submodules',
                                                          message="What submodule do you want?",
                                                          choices=available_submodules)
        else:
            submodule_answer = {"submodules": []}

        return {
            "name": module_friendly_name,
            "config": module_config,
            "submodules": [self.query_module(module.get_submodule(sm), sm) for sm in
                           submodule_answer["submodules"]]

        }

    @staticmethod
    def checkbox(key: str, message: str, choices: list) -> dict:
        """
        Wrapper method to avoid empty responses in checkbox

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
            return ConfigManager.checkbox(key, message, choices)

        return answer

    def set_config(self, config: BenchmarkConfig):
        """
        In case the user supplies a config file this function is used to set the config

        :param config:  Valid config file
        :type config: BenchmarkConfig
        """
        self.config = config

    def load_config(self, app_modules: dict):
        """
        Uses the config to generate all class instances needed to run the benchmark.

        :param app_modules: the list of application modules as specified in the application modules configuration.
        :type app_modules: list of dict
        :rtype: None
        """

        self.application = _get_instance_with_sub_options(app_modules, self.config["application"]["name"])

        self.config["application"].update({"instance": self.application,
                                           "submodules": [ConfigManager.initialize_module_classes(self.application, c)
                                                          for c
                                                          in
                                                          self.config["application"]["submodules"]]})

    @staticmethod
    def initialize_module_classes(parent_module: Core, config: ConfigModule) -> ConfigModule:
        """
        Recursively initializes all instances of the needed modules and its submodules for a given config

        :param parent_module: Class of the parent module
        :type parent_module: Core
        :param config: uninitialized config module
        :type config: ConfigModule
        :return: Config with instances
        :rtype: ConfigModule
        """

        module_instance = parent_module.get_submodule(config["name"])
        config.update({"instance": module_instance,
                       "submodules": [ConfigManager.initialize_module_classes(module_instance, c) for c in
                                      config["submodules"]]}
                      )
        return config

    def get_config(self) -> BenchmarkConfig:
        """
        Returns the config

        :return: Returns the config
        :rtype: BenchmarkConfig
        """
        return self.config

    def get_app(self) -> Application:
        """
        Returns instance of the application

        :return: Instance of the application
        :rtype: Application
        """
        return self.config["application"]["instance"]

    def get_reps(self) -> int:
        """
        Returns number of repetitions specified in config

        :return: Number of repetitions
        :rtype: int
        """
        return self.config["repetitions"]

    def start_create_benchmark_backlog(self) -> list:
        """
        Helper function to kick of the creation of the benchmark backlog

        :return: List with all benchmark items
        :rtype: list
        """
        return ConfigManager.create_benchmark_backlog(self.config["application"])

    @staticmethod
    def create_benchmark_backlog(module: ConfigModule) -> list:
        """
        Recursive function which splits up the loaded config into single benchmark runs.

        :param module: config module
        :type module: any
        :return: List with all benchmark items
        :rtype: list
        """
        # fill dict until end is reached and then add dict to the backlog
        items = []

        if len(module["config"].items()) > 0:
            keys, values = zip(*module["config"].items())
            config_expanded = [dict(zip(keys, v)) for v in itertools.product(*values)]
        else:
            config_expanded = [{}]

        for _, single_config in enumerate(config_expanded):
            # Check if we reached the end of the chain
            if len(module["submodules"]) > 0:
                for submodule in module["submodules"]:
                    for item in ConfigManager.create_benchmark_backlog(submodule):
                        items.append({
                            "name": module["name"],
                            "instance": module["instance"],
                            "config": single_config,
                            "submodule": {
                                submodule["name"]: item
                            }
                        })

            else:
                return [{
                    "name": module["name"],
                    "instance": module["instance"],
                    "config": single_config,
                    "submodule": {
                    }
                }]

        return items

    def save(self, store_dir: str):
        """
        Save the config as a YAML file.

        :param store_dir: directory where the file should be stored
        :type store_dir: str
        :rtype: None
        """
        with open(f"{store_dir}/config.yml", 'w') as filehandler:
            yaml.dump(self.config, filehandler)

    def print(self):
        """
        Prints the config
        :rtype: None
        """
        print(yaml.dump(self.config))

    @staticmethod
    def _query_for_config(param_opts: dict, prefix: str = "") -> dict:
        """
        For a given module config, queries the user in an interactive mode, which of the options he would like to
        include in the final benchmark config.

        :param param_opts: Dictionary containing the options for a parameter including a description
        :type param_opts: dict
        :param prefix: Prefix string, which is attached when interacting with the user
        :type prefix: str
        :return: Dictionary containing the decisions of the user on what to include in the benchmark.
        :rtype: dict
        """
        config = {}
        for key, config_answer in param_opts.items():
            if config_answer.get("if"):
                #
                # support parameter descriptions like
                # "seed": {
                #    "if": {"key":"graph_type", "in" : ["erdos-renyi"]},
                #    ...
                # }
                # meaning that 'seed' only gets displayed if graph_type has been chosen to be 'erdos-renyi'
                # This expects that the referenced parameter has been declared before and is declared to be
                # 'exclusive' so that its value is unique.
                #

                key_in_cond = config_answer.get("if")["key"]
                dependency = param_opts.get(key_in_cond)

                # check configuration is consistent
                consistent = False
                err_msg = None
                if dependency is None:
                    err_msg = f"Inconsistent parameter options: condition references unknown parameter: {key_in_cond}"
                elif not dependency.get('exclusive', False):
                    err_msg = f"Inconsistent parameter options: " \
                              f"condition references non exclusive parameter: {key_in_cond}"
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
                answer = ConfigManager.checkbox(key=key,
                                                message=f"{prefix} {config_answer['description']}",
                                                choices=config_answer['values'])
                values = (answer[key],)
            else:

                choices = [*config_answer['values'], "Custom Input"] if (config_answer.get("custom_input") and
                                                                         config_answer["custom_input"]) \
                    else config_answer['values']

                if config_answer.get("allow_ranges") and config_answer["allow_ranges"]:
                    choices.append("Custom Range")
                answer = ConfigManager.checkbox(key=key,
                                                message=f"{prefix} {config_answer['description']}",
                                                # Add custom_input if it is specified in the parameters
                                                choices=choices)
                values = answer[key]

                if "Custom Input" in values:
                    freetext_answer = inquirer.prompt(
                        [inquirer.Text('custom_input', message=f"What's your custom input for {key}? (No validation of "
                                                               "this input is done!)")])

                    # Replace the freetext placeholder with the user input
                    values.remove("Custom Input")
                    values.append(freetext_answer["custom_input"])

                if "Custom Range" in values:
                    range_answer = inquirer.prompt(
                        [inquirer.Text('start', message=f"What's the start of your range for {key}? (No validation of "
                                                        "this input is done!)"),
                         inquirer.Text('stop', message=f"What's the end of your range for {key}? (No validation of "
                                                       "this input is done!)"),
                         inquirer.Text('step', message=f"What are the steps of your range for {key}? (No validation of "
                                                       "this input is done!)")])

                    values.remove("Custom Range")
                    values.extend(np.arange(float(range_answer["start"]), float(range_answer["stop"]),
                                            float(range_answer["step"])))
                    # Remove possible duplicates
                    values = list(set(values))

            if config_answer.get("postproc"):
                # the value of config_answer.get("postproc") is expected to be callable
                # with each of the user selected values as argument.
                # Note that the stored config file will contain the processed values.
                values = [config_answer["postproc"](v) for v in values]
            config[key] = values

        return config

    def create_tree_figure(self, store_dir: str):
        """
        Visualize the benchmark as a graph, experimental feature!

        :param store_dir: directory where the file should be stored
        :type store_dir: str
        :rtype: None
        """

        graph = nx.DiGraph()

        ConfigManager._create_tree_figure_helper(graph, self.config["application"])

        nx.draw(graph, with_labels=True, pos=nx.spectral_layout(graph), node_shape="s")
        plt.savefig(f"{store_dir}/BenchmarkGraph.png", format="PNG")
        plt.clf()

    @staticmethod
    def _create_tree_figure_helper(graph: nx.Graph, config: ConfigModule):
        """
        Helper for _create_tree_figure that traverses the config recursively

        :param graph: networkx Graph
        :type graph: networkx.Graph
        :param config: benchmark config
        :type config: dict
        :rtype: None
        """

        if config:
            key = config["name"]
            if "submodules" in config and config["submodules"]:
                for submodule in config["submodules"]:
                    graph.add_edge(key, submodule["name"])
                    ConfigManager._create_tree_figure_helper(graph, submodule)
