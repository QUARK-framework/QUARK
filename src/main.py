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
import yaml
import sys
import os

# add the paths before the following imports
# 'noqa E402' tells PyCharm to ignore the pep8 violation "E402: module level import not at top of file" on these lines
install_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(install_dir)

import argparse  # noqa E402
from collections.abc import Iterable  # noqa E402
from typing import Union  # noqa E402
from BenchmarkManager import BenchmarkManager  # noqa E402


def _filter_comments(f: Iterable) -> str:
    """
    Returns the content of the filehandle f, ignoring all lines starting with '#'.

    :param f: the file to be read
    :type f: Iterable
    :return: the file content without comment lines
    :rtype: str
    """
    lines = []
    for l in f:
        if l.strip().startswith("#"):
            continue
        lines.append(l)
    return "".join(lines)


def _expand_paths(j: Union[dict, list], base_dir: str) -> Union[dict, list]:
    """
    Expands the paths given as value of the 'dir' attribute appearing in the QUARK modules
    configuration by joining base_dir with that path.

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
            if attr in ["mappings","solvers","devices"]:
                _expand_paths(j[attr], base_dir)
            elif attr == "dir":
                p = j[attr]
                if not os.path.isabs(p):
                    j[attr] = os.path.join(base_dir, p)
    return j


def main() -> None:
    """
    Main function that triggers the benchmarking process.
    """
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("logger.log"),
            logging.StreamHandler()
        ]
    )

    # May be overridden by using the -m|--modules option
    app_modules = [
        {"name": "PVC", "module": "applications.PVC.PVC"},
        {"name": "SAT", "module": "applications.SAT.SAT"},
        {"name": "TSP", "module": "applications.TSP.TSP"}
    ]

    try:
        benchmark_manager = BenchmarkManager()

        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--config", help="Provide valid config file instead of interactive mode")
        parser.add_argument('-s', '--summarize', nargs='+', help='If you want to summarize multiple experiments',
                            required=False)
        parser.add_argument('-m', '--modules', help="Provide a file listing the modules to be loaded")
        args = parser.parse_args()
        if args.summarize:
            benchmark_manager.summarize_results(args.summarize)
        else:
            if args.modules:
                logging.info(f"load application modules configuration from {args.modules}")
                # preprocess the 'modules' configuration:
                #   + filter comment lines (lines starting with '#')
                #   + replace relative paths by taking them relative to the location of the modules configuration file.
                base_dir = os.path.dirname(args.modules)
                app_modules = _expand_paths(json.loads(_filter_comments(open(args.modules))), base_dir)
            if args.config:
                logging.info(f"Provided config file at {args.config}")
                # Load config
                f = open(args.config)

                # returns JSON object as a dictionary
                benchmark_config = yaml.load(f, Loader=yaml.FullLoader)
            else:
                benchmark_config = benchmark_manager.generate_benchmark_configs(app_modules)

            benchmark_manager.orchestrate_benchmark(benchmark_config, app_modules)
            df = benchmark_manager.load_results()
            benchmark_manager.visualize_results(df)
    except Exception as e:
        logging.error(e)
        raise e


if __name__ == '__main__':
    main()
