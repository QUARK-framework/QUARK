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
import logging
import json
from collections.abc import Iterable
import yaml

from quark.Installer import Installer
from quark.utils import _expand_paths
from quark.utils_mpi import MPIStreamHandler, MPIFileHandler, get_comm

comm = get_comm()

# Add the paths
install_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(install_dir)

# The following line is currently needed for the hybrid jobs repo
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)))


def _filter_comments(file: Iterable) -> str:
    """
    Returns the content of the filehandle, ignoring all lines starting with '#'.

    :param file: file to be read
    :return: file content without comment lines
    """
    lines = []
    for line in file:
        if line.strip().startswith("#"):
            continue
        lines.append(line)
    return "".join(lines)


def setup_logging() -> None:
    """
    Sets up the logging.
    """
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            MPIFileHandler("logging.log"),
            MPIStreamHandler()
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


def start_benchmark_run(config_file: str = None, store_dir: str = None,
                        fail_fast: bool = False) -> None:
    """
    Starts a benchmark run from the code.
    """
    # TODO not reachable? dead code?
    setup_logging()

    # Helper for hybrid jobs
    if not config_file:
        config_file = os.environ["AMZN_BRAKET_HP_FILE"]
    if not store_dir:
        store_dir = os.environ["AMZN_BRAKET_JOB_RESULTS_DIR"]

    # Reads the config file
    with open(config_file, "r") as f:
        benchmark_config = json.load(f)

    benchmark_config = json.loads(benchmark_config["config"])

    from quark.BenchmarkManager import BenchmarkManager  # pylint: disable=C0415
    from quark.ConfigManager import ConfigManager  # pylint: disable=C0415

    config_manager = ConfigManager()
    config_manager.set_config(benchmark_config)

    benchmark_manager = BenchmarkManager(fail_fast=fail_fast)

    # Can be overridden by using the -m|--modules option
    installer = Installer()
    app_modules = installer.get_env(installer.get_active_env())
    benchmark_manager.orchestrate_benchmark(
        config_manager, store_dir=store_dir, app_modules=app_modules
    )


def create_benchmark_parser(parser: argparse.ArgumentParser):
    parser.add_argument("-c", "--config", help="Provide valid config file instead of interactive mode")
    parser.add_argument('-cc', '--createconfig', help='If you want o create a config without executing it',
                        required=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('-s', '--summarize', nargs='+', help='If you want to summarize multiple experiments',
                        required=False)
    parser.add_argument('-m', '--modules', help="Provide a file listing the modules to be loaded")
    parser.add_argument('-rd', '--resume-dir', nargs='?', help='Provide results directory of the job to be resumed')
    parser.add_argument('-ff', '--failfast', help='Flag whether a single failed benchmark run causes QUARK to fail',
                        required=False, action=argparse.BooleanOptionalAction)

    parser.set_defaults(goal='benchmark')


def create_env_parser(parser: argparse.ArgumentParser):
    subparsers = parser.add_subparsers(help='env')

    parser_env = subparsers.add_parser('env', help='When you want to change something about the QUARK module env.')
    parser_env.add_argument('-c', '--configure', help='Configure certain QUARK modules', required=False)
    parser_env.add_argument('-a', '--activate', help='Activate a certain set of modules', required=False)
    parser_env.add_argument('-cm', '--createmoduledb', help='Create module db', required=False,
                            action=argparse.BooleanOptionalAction)
    parser_env.add_argument('-l', '--list', help='List existing environments', required=False,
                            action=argparse.BooleanOptionalAction)
    parser_env.add_argument('-s', '--show', help='Show the content of an env', required=False)
    parser_env.set_defaults(goal='env')


def handle_benchmark_run(args: argparse.Namespace) -> None:
    """
    Handles the different options of a benchmark run.

    :param args: Namespace with the arguments given by the user
    """
    from quark.BenchmarkManager import BenchmarkManager  # pylint: disable=C0415
    from quark.Plotter import Plotter  # pylint: disable=C0415

    benchmark_manager = BenchmarkManager(fail_fast=args.failfast)

    if args.summarize:
        benchmark_manager.summarize_results(args.summarize)
    else:
        from quark.ConfigManager import ConfigManager  # pylint: disable=C0415
        config_manager = ConfigManager()
        if args.modules:
            logging.info(f"Load application modules configuration from {args.modules}")
            # Preprocesses the 'modules' configuration:
            #   + Filters comment lines (lines starting with '#')
            #   + Replaces relative paths by taking them relative to the location of the modules configuration file
            base_dir = os.path.dirname(args.modules)
            with open(args.modules) as filehandler:
                app_modules = _expand_paths(json.loads(
                    _filter_comments(filehandler)), base_dir
                )
        else:
            # Gets current env here
            installer = Installer()
            app_modules = installer.get_env(installer.get_active_env())

        if args.config or args.resume_dir:
            if not args.config:
                args.config = os.path.join(args.resume_dir, "config.yml")
            logging.info(f"Provided config file at {args.config}")
            # Loads config
            with open(args.config) as filehandler:
                # Returns JSON object as a dictionary
                try:
                    benchmark_config = yaml.load(filehandler, Loader=yaml.FullLoader)
                except Exception as e:
                    logging.exception("Problem loading the given config file")
                    raise ValueError("Config file needs to be a valid QUARK YAML Config!") from e

                config_manager.set_config(benchmark_config)
        else:
            config_manager.generate_benchmark_configs(app_modules)

        if args.createconfig:
            logging.info("Selected config is:")
            config_manager.print()
        else:
            interrupted_results_path = None if args.resume_dir is None else os.path.join(
                args.resume_dir, "results.json"
            )
            benchmark_manager.orchestrate_benchmark(
                config_manager, app_modules,
                interrupted_results_path=interrupted_results_path
            )
            comm.Barrier()
            if comm.Get_rank() == 0:
                results = benchmark_manager.load_results()
                Plotter.visualize_results(results, benchmark_manager.store_dir)


def handler_env_run(args: argparse.Namespace) -> None:
    """
    Orchestrates the requests to the QUARK module environment.

    :param args: Namespace with the arguments given by the user
    """
    installer = Installer()
    if args.createmoduledb:
        installer.create_module_db()
    elif args.activate:
        installer.set_active_env(args.activate)
    elif args.configure:
        installer.configure(args.configure)
    elif args.list:
        installer.list_envs()
    elif args.show:
        installer.show(installer.get_env(args.show))


def start() -> None:
    """
    Main function that triggers the benchmarking process
    """
    setup_logging()

    try:
        parser = argparse.ArgumentParser()
        create_benchmark_parser(parser)
        create_env_parser(parser)

        args = parser.parse_args()
        if args.goal == "env":
            handler_env_run(args)

        else:
            handle_benchmark_run(args)

        logging.info(" ============================================================ ")
        logging.info(" ====================  QUARK finished!   ==================== ")
        logging.info(" ============================================================ ")

    except Exception as error:
        logging.error(error)
        raise error
