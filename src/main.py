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

import os
import sys
import argparse
from collections.abc import Iterable
import yaml

from utils import _expand_paths

# add the paths before the following imports
install_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(install_dir)

# The following line is at the moment needed for the hybrid jobs repo
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)))

from BenchmarkManager import *
from ConfigManager import ConfigManager


def _filter_comments(file: Iterable) -> str:
    """
    Returns the content of the filehandle f, ignoring all lines starting with '#'.

    :param file: the file to be read
    :type file: Iterable
    :return: the file content without comment lines
    :rtype: str
    """
    lines = []
    for line in file:
        if line.strip().startswith("#"):
            continue
        lines.append(line)
    return "".join(lines)


def setup_logging() -> None:
    """
    Setup the logging

    :return:
    :rtype: None
    """
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("logging.log"),
            logging.StreamHandler()
        ]
    )

    logging.info(" ============================================================ ")
    logging.info(r"             ___    _   _      _      ____    _  __           ")
    logging.info(r"            / _ \  | | | |    / \    |  _ \  | |/ /           ")
    logging.info(r"           | | | | | | | |   / _ \   | |_) | | ' /            ")
    logging.info(r"           | |_| | | |_| |  / ___ \  |  _ <  | . \            ")
    logging.info(r"            \__\_\  \___/  /_/   \_\ |_| \_\ |_|\_\           ")
    logging.info("                                                              ")
    logging.info(" ============================================================ ")
    logging.info("  A Framework for Quantum Computing Application Benchmarking  ")
    logging.info("                                                              ")
    logging.info("        Licensed under the Apache License, Version 2.0        ")
    logging.info(" ============================================================ ")


def get_default_app_modules() -> List[Dict]:
    """
    Returns the default application modules which should be imported.

    :return: List with the default application modules
    :rtype: List[Dict]
    """
    return [
        {"name": "PVC", "module": "modules.applications.optimization.PVC.PVC"},
        {"name": "SAT", "module": "modules.applications.optimization.SAT.SAT"},
        {"name": "TSP", "module": "modules.applications.optimization.TSP.TSP"}
    ]


def start_benchmark_run(config_file: str = None, store_dir: str = None) -> None:
    """
    Function to start a benchmark run from the code

    :rtype: None
    """

    setup_logging()

    # Helper for Hybrid Jobs
    # TODO Check if this can be done better
    if not config_file:
        config_file = os.environ["AMZN_BRAKET_HP_FILE"]
    if not store_dir:
        store_dir = os.environ["AMZN_BRAKET_JOB_RESULTS_DIR"]

    # Read the config file
    with open(config_file, "r") as f:
        benchmark_config = json.load(f)

    benchmark_config = json.loads(benchmark_config["config"])

    config_manager = ConfigManager()
    config_manager.set_config(benchmark_config)
    benchmark_manager = BenchmarkManager()
    # May be overridden by using the -m|--modules option
    app_modules = get_default_app_modules()
    benchmark_manager.orchestrate_benchmark(config_manager, store_dir=store_dir, app_modules=app_modules)


def main() -> None:
    """
    Main function that triggers the benchmarking process.
    """
    setup_logging()

    try:
        benchmark_manager = BenchmarkManager()

        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--config", help="Provide valid config file instead of interactive mode")
        parser.add_argument('-cc', '--createconfig', help='If you want o create a config without executing it',
                            required=False, action=argparse.BooleanOptionalAction)
        parser.add_argument('-s', '--summarize', nargs='+', help='If you want to summarize multiple experiments',
                            required=False)
        parser.add_argument('-m', '--modules', help="Provide a file listing the modules to be loaded")
        args = parser.parse_args()
        if args.summarize:
            benchmark_manager.summarize_results(args.summarize)
        else:
            config_manager = ConfigManager()
            if args.modules:
                logging.info(f"load application modules configuration from {args.modules}")
                # preprocess the 'modules' configuration:
                #   + filter comment lines (lines starting with '#')
                #   + replace relative paths by taking them relative to the location of the modules configuration file.
                base_dir = os.path.dirname(args.modules)
                with open(args.modules) as filehandler:
                    app_modules = _expand_paths(json.loads(_filter_comments(filehandler)), base_dir)
            else:
                app_modules = get_default_app_modules()

            if args.config:
                logging.info(f"Provided config file at {args.config}")
                # Load config
                with open(args.config) as filehandler:
                    # returns JSON object as a dictionary
                    benchmark_config = yaml.load(filehandler, Loader=yaml.FullLoader)
                    config_manager.set_config(benchmark_config)
            else:
                config_manager.generate_benchmark_configs(app_modules)

            if args.createconfig:
                logging.info("Selected config is:")
                config_manager.print()
            else:
                benchmark_manager.orchestrate_benchmark(config_manager, app_modules)
                results = benchmark_manager.load_results()
                benchmark_manager.vizualize_results(results, benchmark_manager.store_dir)

            logging.info(" ============================================================ ")
            logging.info(" ====================  QUARK finished!   ==================== ")
            logging.info(" ============================================================ ")

    except Exception as error:
        logging.error(error)
        raise error


if __name__ == '__main__':
    main()
