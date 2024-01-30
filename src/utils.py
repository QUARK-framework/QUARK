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

import importlib
import logging
import os
import subprocess
import sys
import time
from typing import Union

import inquirer


def _get_instance_with_sub_options(options: list[dict], name: str) -> any:
    """
    Creates an instance of the QUARK module identified by class_name

    :param options: Section of the QUARK module configuration including the submodules' information.
    :type options: list of dict
    :param name: name of the QUARK component to be initialized
    :type name: str
    :return: New instance of the QUARK module
    :rtype: any
    """
    for opt in options:
        if name != opt["name"]:
            continue
        class_name = opt.get("class", name)
        clazz = _import_class(opt["module"], class_name, opt.get("dir"))
        sub_options = None
        if "submodules" in opt:
            sub_options = opt["submodules"]

        # In case the class requires some arguments in its constructor they can be defined in the "args" dict
        if "args" in opt and opt["args"]:
            instance = clazz(**opt["args"])
        else:
            instance = clazz()

        # _get_instance_with_sub_options is mostly called when using the --modules option, so it makes sense to also
        # save the git revision of the given module, since it can be in a different git

        # Directory of this file
        utils_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        module_dir = os.path.dirname(sys.modules[opt["module"]].__file__)

        # Only log git revision number, if the module is not in the same directory as the utils file
        if not os.path.commonprefix([utils_dir, module_dir]) == utils_dir:
            git_revision_number, git_uncommitted_changes = get_git_revision(module_dir)

            instance.metrics.add_metric_batch({
                "module_git_revision_number": git_revision_number,
                "module_git_uncommitted_changes": git_uncommitted_changes})

        # sub_options inherits 'dir'
        if sub_options and "dir" in opt:
            for sub_opt in sub_options:
                if "dir" not in sub_opt:
                    sub_opt["dir"] = opt["dir"]

        instance.sub_options = sub_options
        return instance
    logging.error(f"{name} not found in {options}")
    raise ValueError(f"{name} not found in {options}")


def _import_class(module_path: str, class_name: str, base_dir: str = None) -> type:
    """
    Helper function which allows to replace hard-coded imports of the form
    'import MyClass from path.to.mypkg' by calling _import_class('path.to.mypkg', 'MyClass').
    If base_dir is specified, its value will be added to the python search path,
    unless it's already contained in it.

    :param module_path: Python module path of the module containing the class to be imported
    :type module_path: str
    :param class_name: Name of the class to be imported
    :type class_name: str
    :return: Imported class object
    :rtype: type
    """

    # Make sure that base_dir is in the search path.
    # Otherwise, the module imported here might not find its libraries.
    if base_dir is not None and base_dir not in sys.path:
        logging.info(f"Append to sys.path: {base_dir}")
        sys.path.append(base_dir)
    logging.info(f"Import module {module_path}")
    module = importlib.import_module(module_path)
    return vars(module)[class_name]


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
        return checkbox(key, message, choices)

    return answer


def get_git_revision(git_dir: str) -> (str, str):
    """
    Collects git revision number and checks if there are uncommitted changes to allow user to analyze which
    codebase was used

    :param git_dir: Directory of the git repository
    :type git_dir: str
    :return: Tuple with git_revision_number, git_uncommitted_changes
    :rtype: (str, str)
    """
    try:
        # '-C', git_dir ensures that the following commands also work when QUARK is started from other working
        # directories
        git_revision_number = subprocess.check_output(['git', '-C', git_dir, 'rev-parse', 'HEAD']).decode(
            'ascii').strip()
        git_uncommitted_changes = bool(subprocess.check_output(
            ['git', '-C', git_dir, 'status', '--porcelain', '--untracked-files=no']).decode(
            'ascii').strip())

        logging.info(
            f"Codebase is based on revision {git_revision_number} and has "
            f"{'some' if git_uncommitted_changes else 'no'} uncommitted changes")
    except Exception as e:
        logging.warning(f"Logging of git revision number not possible because of: {e}")
        git_revision_number = "unknown"
        git_uncommitted_changes = "unknown"

    return git_revision_number, git_uncommitted_changes


def _expand_paths(j: Union[dict, list], base_dir: str) -> Union[dict, list]:
    """
    Expands the paths given as value of the 'dir' attribute appearing in the QUARK modules
    configuration by joining base_dir with that path

    :param j: the json to be adapted - expected to be a QUARK modules configuration or a part of it
    :type j: dict|list
    :param base_dir: the base directory to be used for path expansion
    :type base_dir: str
    :return: the adapted json
    :rtype: dict|list
    """
    assert type(j) in [dict, list], f"unexpected type:{type(j)}"
    if type(j) == list:
        for entry in j:
            _expand_paths(entry, base_dir)
    else:
        for attr in j:
            if type(j[attr]) == "submodules":
                _expand_paths(j[attr], base_dir)
            elif attr == "dir":
                p = j[attr]
                if not os.path.isabs(p):
                    j[attr] = os.path.join(base_dir, p)
    return j


def start_time_measurement() -> float:
    """
    Starts a time measurement

    :return: Starting point
    :rtype: float
    """
    return time.perf_counter()


def end_time_measurement(start: float) -> float:
    """
    Returns the result of the time measurement in milliseconds

    :param start: Starting point for the measurement
    :type start: float
    :return: Time elapsed in ms
    :rtype: float
    """
    end = time.perf_counter()
    return round((end - start) * 1000, 3)


def quark_stop_watch(position: int = None):
    """
    Usage as decorator to measure time, eg:
    ```
    @quark_stop_watch()
    def run(input_data,...):
        return processed_data
    ```
    results in valid:    
    ```
    processed_data, time_to_process = run(input,...)
    ```
    If the return value of the decorated function is of type tuple,
    the optional parameter `position` can be used to specify the position at which the
    measured time is to be inserted in the returned tuple.

    :param position: the position at which to insert the time
    :type position: int
    """
    def wrap(func):
        def wrapper(*args, **kwargs):
            start = start_time_measurement()
            return_value = func(*args, **kwargs)
            duration = end_time_measurement(start)
            if position is not None and isinstance(return_value, tuple):
                rv = list(return_value)
                rv.insert(position, duration)
                return tuple(rv)
            return return_value, duration
        return wrapper
    return wrap
