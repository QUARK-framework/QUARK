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

import logging
import json
import os
import time
from pathlib import Path
import inspect

import yaml
from packaging import version
import inquirer

from quark.modules.Core import Core
from quark.utils import _get_instance_with_sub_options, get_git_revision, checkbox


class Installer:
    """
    Installer class that can be used by the user to install certain QUARK modules and also return the required Python
    packages for the demanded modules.
    """

    def __init__(self):
        self.quark_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
        self.settings_dir = f"{self.quark_dir}/.settings"
        self.envs_dir = f"{self.settings_dir}/envs"
        self.python_version = "3.12.7"
        self.pip_version = "23.0"
        self.default_app_modules = [
            {"name": "PVC", "class": "PVC", "module": "modules.applications.optimization.PVC.PVC"},
            {"name": "SAT", "class": "SAT", "module": "modules.applications.optimization.SAT.SAT"},
            {"name": "TSP", "class": "TSP", "module": "modules.applications.optimization.TSP.TSP"},
            {"name": "ACL", "class": "ACL", "module": "modules.applications.optimization.ACL.ACL"},
            {"name": "MIS", "class": "MIS", "module": "modules.applications.optimization.MIS.MIS"},
            {"name": "SCP", "class": "SCP", "module": "modules.applications.optimization.SCP.SCP"},
            {"name": "GenerativeModeling", "class": "GenerativeModeling",
             "module": "modules.applications.qml.generative_modeling.GenerativeModeling"}
        ]

        self.core_requirements = [
            {"name": "seaborn", "version": "0.13.2"},
            {"name": "networkx", "version": "3.4.2"},
            {"name": "inquirer", "version": "3.4.0"},
            {"name": "packaging", "version": "24.2"},
            {"name": "pyyaml", "version": "6.0.2"},
            {"name": "typing-extensions", "version": "4.12.2"},
            {"name": "sphinx", "version": "8.1.3"},
            {"name": "sphinx-rtd-theme", "version": "3.0.2"},
        ]
        Path(self.envs_dir).mkdir(parents=True, exist_ok=True)

    def configure(self, env_name="default") -> None:
        """
        Configures a new QUARK environment or overwrites an existing one.

        :param env_name: Name of the env to configure
        """
        configured_envs = self.check_for_configs()

        if env_name in configured_envs:
            answer_continue = inquirer.prompt([
                inquirer.List("continue",
                              message=f"{env_name} found in the list of existing QUARK module environment, are you"
                              f" sure you want to overwrite it?",
                              choices=["Yes", "No"], )])["continue"]

            if answer_continue.lower() == "no":
                logging.info("Exiting configuration")
                return

        chosen_config_type = inquirer.prompt([
            inquirer.List("config",
                          message="Do you want to use the default configuration or a custom environment?",
                          choices=["Default", "Custom"])])["config"]
        logging.info(f"You chose {chosen_config_type}")

        module_db = self.get_module_db()

        if chosen_config_type == "Default":
            self.save_env(module_db, env_name)
        elif chosen_config_type == "Custom":
            module_db = self.start_query_user(module_db)
            self.save_env(module_db, env_name)

        requirements = self.collect_requirements(module_db["modules"])
        activate_requirements = checkbox("requirements", "Should we create an package file, if yes for "
                                                         "which package manager?",
                                         ["Conda", "PIP", "Print it here"])["requirements"]

        if "Conda" in activate_requirements:
            self.create_conda_file(requirements, env_name)
        if "PIP" in activate_requirements:
            self.create_req_file(requirements, env_name)
        if "Print it here" in activate_requirements:
            logging.info("Please install:")
            for p, v in requirements.items():
                logging.info(f"  -  {p}{': ' + v[0] if v else ''}")

        activate_answer = inquirer.prompt([
            inquirer.List("activate",
                          message="Do you want to activate the QUARK module environment?",
                          choices=["Yes", "No"])])["activate"]

        if activate_answer == "Yes":
            self.set_active_env(env_name)

    def check_for_configs(self) -> list:
        """
        Checks if QUARK is already configured and if yes, which environments.

        :return: Returns the configured QUARK envs in a list
        """
        return list(p.stem for p in Path(self.envs_dir).glob("*.json"))

    def set_active_env(self, name: str) -> None:
        """
        Sets the active env to active_env.json.

        :param name: Name of the env
        """
        self._check_if_env_exists(name)
        with open(f"{self.settings_dir}/active_env.json", "w") as jsonFile:
            json.dump({"name": name}, jsonFile, indent=2)

        logging.info(f"Set active QUARK module environment to {name}")

    def check_active_env(self) -> bool:
        """
        Checks if .settings/active_env.json exists.

        :return: True if active_env.json exists
        """
        return Path(f"{self.settings_dir}/active_env.json").is_file()

    def get_active_env(self) -> str:
        """
        Returns the current active environment.

        :return: Returns the name of the active env
        """
        # if not self.check_active_env():
        #     logging.warning("No active QUARK module environment found, using default")
        #     module_db = self.get_module_db()
        #     self.save_env(module_db, "default")
        #     self.set_active_env("default")

        # with open(f"{self.settings_dir}/active_env.json", "r") as filehandler:
        #     env = json.load(filehandler)
        #     return env["name"]

        return ""

    def get_env(self, name: str) -> list[dict]:
        """
        Loads the env from file and returns it.

        :param name: Name of the env
        :return: Returns the modules of the env
        """
        # file = f"{self.envs_dir}/{name}.json"
        # self._check_if_env_exists(name)

        # with open(file, "r") as filehandler:
        #     env = json.load(filehandler)
        #     logging.info(f"Getting {name} QUARK module environment")
        #     module_db_build_number = self.get_module_db_build_number()
        #     if env["build_number"] < module_db_build_number:
        #         logging.warning(f"You QUARK module env is based on an outdated build version of the module database "
        #                         f"(BUILD NUMBER {env['build_number']}). The current module database (BUILD NUMBER "
        #                         f"{module_db_build_number}) might bring new features. You should think about "
        #                         f"updating your environment!")

        #     return env["modules"]
        return self.default_app_modules

    def _check_if_env_exists(self, name: str) -> str:
        """
        Checks if a given env exists, returns the location of the associated JSON file and raises an error otherwise.

        :param name: Name of the env
        :return: Returns location of the JSON file associated with the env if it exists
        """
        file = f"{self.envs_dir}/{name}.json"
        if not Path(file).is_file():
            raise ValueError(f"QUARK environment {name} could not be found!")
        return file

    def save_env(self, env: dict, name: str) -> None:
        """
        Saves a created env to a file with the name of choice.

        :param env: Env which should be saved
        :param name: Name of the env
        """
        with open(f"{self.envs_dir}/{name}.json", "w") as jsonFile:
            json.dump(env, jsonFile, indent=2)

        logging.info(f"Saved {name} QUARK module environment.")

    def start_query_user(self, module_db: dict) -> dict:
        """
        Queries the user which applications and submodules to include.

        :param module_db: module_db file
        :return: Returns the module_db with selected (sub)modules
        """
        answer_apps = checkbox("apps", "Which application would you like to include?",
                               [m["name"] for m in module_db["modules"]])["apps"]

        module_db["modules"] = [x for x in module_db["modules"] if x["name"] in answer_apps]

        for idx, entry in enumerate(module_db["modules"]):
            self.query_user(module_db["modules"][idx], entry["name"])

        return module_db

    def query_user(self, submodules: dict, name: str) -> None:
        """
        Queries the user which submodules to include

        :param submodules: Submodules for the module
        :param name: Name of the module
        """
        if submodules["submodules"]:
            answer_submodules = \
                checkbox("submodules", f"Which submodule would you like to include for {name}?",
                         [m["name"] for m in submodules["submodules"]])["submodules"]

            submodules["submodules"] = [x for x in submodules["submodules"] if x["name"] in answer_submodules]
            for idx, entry in enumerate(submodules["submodules"]):
                self.query_user(submodules["submodules"][idx], entry["name"])

    def get_module_db(self) -> dict:
        """
        Returns the module database that contains all module possibilities.

        :return: Module Database
        """
        return {
            "build_number": 1,
            "build_date": time.strftime("%d-%m-%Y %H:%M:%S", time.gmtime()),
            "git_revision_number": 1,
            "modules": self.default_app_modules
        }

    def create_module_db(self) -> None:
        """
        Creates module database by automatically going through the available submodules for each module.
        """
        logging.info("Creating Module Database")

        # TODO Helper to skip constructor in Braket.py. Should probably be changed in the future!
        os.environ["SKIP_INIT"] = "True"

        module_db_modules: list[dict] = self.default_app_modules

        for idx, app in enumerate(module_db_modules):
            logging.info(f"Processing {app['name']}")
            app_instance = _get_instance_with_sub_options(module_db_modules, app["name"])
            module_db_modules[idx]["submodules"] = [
                Installer._create_module_db_helper(app_instance.get_submodule(sm), sm) for
                sm in app_instance.submodule_options]
            module_db_modules[idx]["requirements"] = app_instance.get_requirements()

        git_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", )
        git_revision_number, _ = get_git_revision(git_dir)
        module_db = {
            "build_number": self.get_module_db_build_number() + 1,
            "build_date": time.strftime("%d-%m-%Y %H:%M:%S", time.gmtime()),
            "git_revision_number": git_revision_number,
            "modules": module_db_modules
        }

        # loop over first level and add to file
        with open(f"{self.settings_dir}/module_db.json", "w") as jsonFile:
            json.dump(module_db, jsonFile, indent=2)

        requirements = self.collect_requirements(module_db_modules)
        self.create_req_file(requirements, "full", self.settings_dir)

        logging.info("Created module_db file")

    @staticmethod
    def _create_module_db_helper(module: Core, name: str) -> dict:
        """
        Recursive helper function for create_module_db.

        :param module: Module instance
        :param name: Name of the module
        :return: Module dict
        """
        return {
            "name": name,
            "class": module.__class__.__name__,
            "args": {k: v for k, v in module.__dict__.items() if k in
                     inspect.signature(module.__init__).parameters.keys()},
            "module": module.__module__,
            "requirements": module.get_requirements(),
            "submodules": [Installer._create_module_db_helper(module.get_submodule(sm), sm) for sm in
                           module.submodule_options]
        }

    def get_module_db_build_number(self) -> int:
        """
        Returns the build number of the module_db.

        :return: Returns the build number of the module_db if it exists, otherwise 0
        """
        if Path(f"{self.settings_dir}/module_db.json").is_file():
            module_db = self.get_module_db()
            return module_db["build_number"]
        else:
            return 0

    def collect_requirements(self, env: list[dict]) -> dict:
        """
        Collects requirements of the different modules in the given env file.

        :param env: Environment configuration
        :return: Collected requirements
        """
        requirements: list[dict] = self.core_requirements
        for app in env:
            requirements.extend(Installer._collect_requirements_helper(app))

        # Counts duplicates and checks if any version conflicts occur
        merged_requirements: dict = {}
        for req in requirements:
            if req["name"] in merged_requirements:
                # Checks if the specific version is already there, if so the req is skipped
                if merged_requirements[req["name"]] and ("version" in req and req["version"] not in
                                                         merged_requirements[req["name"]]):
                    merged_requirements[req["name"]].append(req["version"])

            else:
                # Sometimes there is no specific version required, then the "version" field is missing
                merged_requirements[req["name"]] = [req["version"]] if "version" in req else []

        for k, v in merged_requirements.items():
            if len(v) > 1:
                # If there are multiple different versions, the latest version is selected
                newest_version = sorted(v, key=lambda x: version.Version(x))[-1]  # pylint: disable=W0108
                merged_requirements[k] = [newest_version]
                logging.warning(f"Different version requirements for {k}: {v}. Will try to take the newer version "
                                f"{newest_version}, but this might cause problems at QUARK runtime!")

        return merged_requirements

    @staticmethod
    def _collect_requirements_helper(module: dict) -> list[dict]:
        """
        Helper function for collect_requirements_helper that recursively checks modules for requirements.

        :param module: Module dict
        :return: List of dicts with the requirements
        """
        requirements = module["requirements"]
        for submodule in module["submodules"]:
            requirements.extend(Installer._collect_requirements_helper(submodule))

        return requirements

    def create_conda_file(self, requirements: dict, name: str, directory: str = None) -> None:
        """
        Creates conda yaml file based on the requirements.

        :param requirements: Collected requirements
        :param name: Name of the conda env
        :param directory: Directory where the file should be saved. If None self.envs_dir will be taken.
        """
        if directory is None:
            directory = self.envs_dir
        conda_data = {
            "name": name,
            "channels": ["defaults"],
            "dependencies": [
                f"python={self.python_version}",
                f"pip={self.pip_version}",
                {"pip": [(f"{k}=={v[0]}" if v else k) for k, v in requirements.items()]}

            ]
        }
        with open(f"{directory}/conda_{name}.yml", "w") as filehandler:
            yaml.dump(conda_data, filehandler)

        logging.info("Saved conda env file, if you like to install it run:")
        logging.info(f"conda env create -f {directory}/conda_{name}.yml")

    def create_req_file(self, requirements: dict, name: str, directory: str = None) -> None:
        """
        Creates pip txt file based on the requirements.

        :param requirements: Collected requirements
        :param name: Name of the env
        :param directory: Directory where the file should be saved. If None self.envs_dir will be taken.
        """
        if directory is None:
            directory = self.envs_dir
        with open(f"{directory}/requirements_{name}.txt", "w") as filehandler:
            for k, v in requirements.items():
                filehandler.write(f"{k}=={v[0]}" if v else k)
                filehandler.write("\n")

        logging.info("Saved pip txt file, if you like to install it run:")
        logging.info(f"pip install -r {directory}/requirements_{name}.txt")

    def list_envs(self) -> None:
        """
        List all existing environments.
        """
        logging.info("Existing environments:")
        for env in self.check_for_configs():
            logging.info(f"  -  {env}")

    @staticmethod
    def show(env: list[dict]) -> None:
        """
        Visualize the env.

        :param env: Environment configuration
        """
        space = "    "
        branch = "|   "
        connector = "|-- "
        leaf = ">-- "

        def tree(modules: list[dict], prefix: str = ""):
            """
             A recursive function that generates a tree from the modules.
             This function is based on https://stackoverflow.com/a/59109706, but modified to the needs here.

            :param modules: Modules list
            :param prefix: Prefix for the indentation
            :return: Generator yielding formatted lines of the environment tree
            """
            # Modules in the middle/beginning get a |--, the final leaf >--
            pointers = [connector] * (len(modules) - 1) + [leaf]
            for pointer, module in zip(pointers, modules):
                yield prefix + pointer + module["name"]

                if module["submodules"]:
                    # If the module has any submodules
                    extension = branch if pointer == connector else space
                    # Check if we are at the end of the tree
                    yield from tree(module["submodules"], prefix=prefix + extension)

        logging.info("Content of the environment:")
        for module in tree(env):
            logging.info(module)
