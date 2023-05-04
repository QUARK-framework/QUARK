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

import yaml
from packaging import version
import inquirer

from ConfigManager import ConfigManager
from modules.Core import Core
from utils import _get_instance_with_sub_options, get_git_revision


class Installer:
    """
    Installer class that can be used by the user to install certain QUARK modules and also return the needed python
    packages for the wanted modules
    """

    def __init__(self):
        self.quark_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", )
        self.settings_dir = f"{self.quark_dir}/.settings"
        self.envs_dir = f"{self.settings_dir}/envs"
        self.python_version = "3.9.16"
        self.pip_version = "23.0"
        self.default_app_modules = [
            {"name": "PVC", "module": "modules.applications.optimization.PVC.PVC"},
            {"name": "SAT", "module": "modules.applications.optimization.SAT.SAT"},
            {"name": "TSP", "module": "modules.applications.optimization.TSP.TSP"}
        ]

        self.core_requirements = [
            {"name": "seaborn", "version": "0.12.2"},
            {"name": "pandas", "version": "1.5.3"},
            {"name": "networkx", "version": "2.8.8"},
            {"name": "inquirer", "version": "3.1.2"},
            {"name": "numpy", "version": "1.24.1"},
            {"name": "pyyaml", "version": "6.0"}
        ]
        Path(self.envs_dir).mkdir(parents=True, exist_ok=True)

    def install(self, env_name="default") -> None:
        """
        Install a new QUARK environment or overwrite an existing one

        :param env_name: Name of the env to install
        :type env_name: str
        :return:
        :rtype: None
        """

        installed_envs = self.check_for_installs()

        if env_name in installed_envs:
            answer_continue = inquirer.prompt([
                inquirer.List('continue',
                              message=f" {env_name} found in the existing QUARK module environment, are you sure you "
                                      f"want to overwrite it?",
                              choices=['Yes', 'No'], )])["continue"]

            if answer_continue == "No":
                logging.info("Exiting install")
                return

        chosen_install_type = inquirer.prompt([
            inquirer.List('installs',
                          message=f"What do you want to install?",
                          choices=['Default', 'Custom'],
                          )])["installs"]
        logging.info(f"You chose {chosen_install_type}")

        module_db = self.get_module_db()

        if chosen_install_type == "Default":
            self.save_env(module_db, env_name)
        elif chosen_install_type == "Custom":
            module_db = self.start_query_user(module_db)
            self.save_env(module_db, env_name)

        requirements = self.collect_requirements(module_db["modules"])
        activate_requirements = ConfigManager.checkbox("requirements", "Should we create an package file, if yes for "
                                                                       "which package manager?",
                                                       ["Conda", "PIP", "Print it here"])["requirements"]

        if "Conda" in activate_requirements:
            self.create_conda_file(requirements, env_name)
        if "PIP" in activate_requirements:
            self.create_req_file(requirements, env_name)
        if "Print it here" in activate_requirements:
            logging.info("Please install:")
            [logging.info(f"  -  {p}{': ' + v[0] if v else ''}") for p, v in requirements.items()]

        activate_answer = inquirer.prompt([
            inquirer.List('activate',
                          message=f"Do you want to activate the QUARK module environment?",
                          choices=["Yes", "No"],
                          )])["activate"]

        if activate_answer == "Yes":
            self.set_active_env(env_name)

    def check_for_installs(self) -> list:
        """
        Check if QUARK was already install and if yes, which environments

        :return:
        :rtype:
        """
        return list(p.stem for p in Path(self.envs_dir).glob("*.json"))

    def set_active_env(self, name: str) -> None:
        """
        Set active env to active_env.json

        :param name: Name of the env
        :type name: str
        :return:
        :rtype: None
        """
        self._check_if_env_exists(name)
        with open(f"{self.settings_dir}/active_env.json", "w") as jsonFile:
            data = {"name": name}
            json.dump(data, jsonFile, indent=2)

        logging.info(f"Set active QUARK module environment to {name}")

    def check_active_env(self) -> bool:
        """
        Check if .settings/active_env.json exists

        :return: True if active_env.json exists
        :rtype: bool
        """
        return Path(f"{self.settings_dir}/active_env.json").is_file()

    def get_active_env(self) -> str:
        """
        Returns the current active environment

        :return: Returns the name of the active env
        :rtype: str
        """
        if not self.check_active_env():
            logging.warning("Not active QUARK module environment found, using default")
            module_db = self.get_module_db()
            self.save_env(module_db, "default")
            self.set_active_env("default")

        with open(f"{self.settings_dir}/active_env.json", "r") as filehandler:
            env = json.load(filehandler)
            return env["name"]

    def get_env(self, name: str) -> list[dict]:
        """
        Loads the env from file and returns it

        :param name: The name of the env
        :type name: dict
        :return: Returns the modules of the env
        :rtype: list[dict]
        """
        file = f"{self.envs_dir}/{name}.json"
        self._check_if_env_exists(name)

        with open(file, 'r') as filehandler:
            env = json.load(filehandler)
            logging.info(f"Getting {name} env")
            module_db_build_number = self.get_module_db_build_number()
            if env["build_number"] < module_db_build_number:
                logging.warning(f"You QUARK module env is based on an outdated build version of the module database "
                                f"(BUILD NUMBER {env['build_number']}). The current module database (BUILD NUMBER {module_db_build_number}) might "
                                f"bring new features. You should think about updating your environment!")

            return env["modules"]

    def _check_if_env_exists(self, name):
        file = f"{self.envs_dir}/{name}.json"
        if not Path(file).is_file():
            raise ValueError(f"QUARK environment {name} could not be found!")
        return file

    def save_env(self, env: dict, name: str) -> None:
        """
        Save a created env to a file with the name of choice

        :param env: Env which should be saved
        :type env: dict
        :param name: Name of the env
        :type name: str
        :return:
        :rtype: None
        """

        with open(f"{self.envs_dir}/{name}.json", "w") as jsonFile:
            json.dump(env, jsonFile, indent=2)

        logging.info(f"Saved {name} QUARK module environment environment.")

    def start_query_user(self, module_db: dict) -> dict:
        """
        Query user which applications and submodules to include

        :param module_db: module db file
        :type module_db: dict
        :return: returns the created env
        :rtype: dict
        """

        answer_apps = ConfigManager.checkbox("apps", "Which application would you like to include",
                                             [m["name"] for m in module_db["modules"]])["apps"]

        module_db["modules"] = [x for x in module_db["modules"] if x["name"] in answer_apps]

        for idx, entry in enumerate(module_db["modules"]):
            self.query_user(module_db["modules"][idx], entry["name"])

        return module_db

    def query_user(self, submodules: dict, name: str) -> None:
        """
        Query user which modules should be included

        :param submodules: the submodules for the module
        :type submodules: dict
        :param name: Name of the module
        :type name: str
        :return:
        :rtype: None
        """

        if submodules["submodules"]:
            answer_submodules = \
                ConfigManager.checkbox("submodules", f"Which submodule would you like to include for {name}",
                                       [m["name"] for m in submodules["submodules"]])["submodules"]

            submodules["submodules"] = [x for x in submodules["submodules"] if x["name"] in answer_submodules]
            for idx, entry in enumerate(submodules["submodules"]):
                self.query_user(submodules["submodules"][idx], entry["name"])

    def get_module_db(self) -> dict:
        """
        Returns module database containing the full module possibilities

        :return: Module Database
        :rtype: dict
        """
        with open(f"{self.settings_dir}/module_db.json", "r") as filehandler:
            return json.load(filehandler)

    def create_module_db(self) -> None:
        """
        Creates module database by automatically going through the available submodules for each module

        :return:
        :rtype: None
        """
        logging.info("Creating Module Database")

        # TODO Helper to skip constructor in Braket.py. Should be changed in the future!
        os.environ['SKIP_INIT'] = "True"

        module_db_modules = self.default_app_modules

        for idx, app in enumerate(module_db_modules):
            logging.info(f"Processing {app['name']}")
            app_instance = _get_instance_with_sub_options(module_db_modules, app["name"])
            module_db_modules[idx]["submodules"] = [
                Installer._create_module_db_helper(app_instance.get_submodule(sm), sm) for
                sm in app_instance.submodule_options]
            module_db_modules[idx]["requirements"] = app_instance.get_requirements()

        git_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", )
        git_revision_number, git_uncommitted_changes = get_git_revision(git_dir)
        module_db = {
            "build_number": self.get_module_db_build_number() + 1,
            "build_date": time.strftime("%d-%m-%Y %H:%M:%S", time.gmtime()),
            "git_revision_number": git_revision_number,
            "modules": module_db_modules
        }

        # loop over first level and add to file
        with open(f"{self.settings_dir}/module_db.json", "w") as jsonFile:
            json.dump(module_db, jsonFile, indent=2)

        logging.info("Created module_db file")

    @staticmethod
    def _create_module_db_helper(module: Core, name: str) -> dict:
        """
        Recursive helper function for create_module_db

        :param module: module
        :type module: Core
        :param name: Name of the module
        :type name: str
        :return: module dict
        :rtype: dict
        """

        return {
            "name": name,
            "module": module.__module__,
            "requirements": module.get_requirements(),
            "submodules": [Installer._create_module_db_helper(module.get_submodule(sm), sm) for sm in
                           module.submodule_options]
        }

    def get_module_db_build_number(self) -> int:
        """
        Return the build number from the module_db

        :return: build number of module_db
        :rtype: int
        """

        if Path(f"{self.settings_dir}/module_db.json").is_file():
            module_db = self.get_module_db()
            return module_db["build_number"]
        else:
            return 0

    def collect_requirements(self, env: dict) -> dict:
        """
        Collect requirements of the different modules in the given env file

        :param env: environment file
        :type env: dict
        :return: Collected requirements
        :rtype: dict
        """

        requirements: list[dict] = self.core_requirements

        [requirements.extend(Installer._collect_requirements_helper(app)) for app in env]

        # Let`s see how many duplicates we have and if we have any version conflicts
        merged_requirements: dict = dict()
        for req in requirements:
            if req["name"] in merged_requirements:
                # Check if the specific version if already there, then we skip it
                if merged_requirements[req["name"]] and ("version" in req and req["version"] not in
                                                         merged_requirements[req["name"]]):
                    merged_requirements[req["name"]].append(req["version"])

            else:
                # Sometimes there is no specific version required, then the "version" field is missing
                merged_requirements[req["name"]] = [req["version"]] if "version" in req else []

        for k, v in merged_requirements.items():
            if len(v) > 1:
                # If there are multiple different version
                newest_version = sorted(v, key=lambda x: version.Version(x))[-1]
                merged_requirements[k] = [newest_version]
                logging.warning(f"Different version requirements for {k}: {v}. Will try to take the newer version "
                                f"{newest_version}, but this might cause problems at QUARK runtime!")

        return merged_requirements

    @staticmethod
    def _collect_requirements_helper(module: dict) -> list[dict]:
        """
        Helper function for collect_requirements_helper that recursively checks modules for requirements

        :param module: module dict
        :type module: dict
        :return: List of dicts with the requirements
        :rtype: list[dict]
        """

        requirements = module["requirements"]
        for submodule in module["submodules"]:
            requirements.extend(Installer._collect_requirements_helper(submodule))

        return requirements

    def create_conda_file(self, requirements: dict, name: str) -> None:
        """
        Create Conda yaml file based on the requirements

        :param requirements: collected requirements
        :type requirements: dict
        :param name: Name of the conda env
        :type name: str
        :return:
        :rtype: None
        """
        conda_data = {
            "name": name,
            "channels": ["defaults"],
            "dependencies": [
                f"python={self.python_version}",
                f"pip={self.pip_version}",
                {"pip": [(f"{k}=={v[0]}" if v else k) for k, v in requirements.items()]}

            ]
        }
        with open(f"{self.envs_dir}/conda_{name}.yml", 'w') as filehandler:
            yaml.dump(conda_data, filehandler)

        logging.info(f"Saved conda env file, if you like to install it run:")
        logging.info(f"conda env create -f {self.envs_dir}/conda_{name}.yml")

    def create_req_file(self, requirements: dict, name: str) -> None:
        """
        Create pip txt file based on the requirements

        :param requirements: collected requirements
        :type requirements: dict
        :param name: Name of the  env
        :type name: str
        :return:
        :rtype: None
        """
        with open(f"{self.envs_dir}/requirements_{name}.txt", "w") as filehandler:
            for k, v in requirements.items():
                filehandler.write(f"{k}=={v[0]}" if v else k)
                filehandler.write("\n")

        logging.info(f"Saved pip txt file, if you like to install it run:")
        logging.info(f"pip install -r  {self.envs_dir}/requirements_{name}.txt")

    def list_envs(self) -> None:
        """
        List all existing environments

        :return:
        :rtype: None
        """

        logging.info("Existing environments:")
        [logging.info(f"  -  {env}") for env in self.check_for_installs()]

    @staticmethod
    def show(env: list[dict]) -> None:
        """
        Visualize the env

        :param env: environment
        :type env: list[dict]
        :return:
        :rtype: None
        """
        space = "    "
        branch = "│   "
        connector = "├── "
        leaf = "└── "

        def tree(modules: list[dict], prefix: str = ''):
            """
             A recursive function that generated a tree from the modules
             This function is based on https://stackoverflow.com/a/59109706, but modified to the needs here
             # TODO check this

            :param modules: Modules
            :type modules: list[dict
            :param prefix: Prefix for the indentation
            :type prefix: str
            :return:
            :rtype:
            """

            # Modules in the middle/beginning get a ├──, the final leaf └──
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
