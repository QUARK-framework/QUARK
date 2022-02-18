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
import config
import yaml
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from BenchmarkManager import BenchmarkManager


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

    try:
        benchmark_manager = BenchmarkManager()

        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--config", help="Provide valid config file instead of interactive mode")
        parser.add_argument('-s', '--summarize', nargs='+', help='If you want to summarize multiple experiments',
                            required=False)
        args = parser.parse_args()
        benchmark_config = None
        if args.summarize:
            benchmark_manager.summarize_results(args.summarize)
        else:
            if args.config:
                logging.info(f"Provided config file at {args.config}")
                # Load config
                f = open(args.config)

                # returns JSON object as a dictionary
                benchmark_config = yaml.load(f, Loader=yaml.FullLoader)
            else:
                benchmark_config = benchmark_manager.generate_benchmark_configs()

            benchmark_manager.orchestrate_benchmark(benchmark_config)
            df = benchmark_manager.load_results()
            benchmark_manager.vizualize_results(df)
    except Exception as e:
        logging.error(e)
        raise e


if __name__ == '__main__':
    main()
