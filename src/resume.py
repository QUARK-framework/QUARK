import argparse
import glob
import logging

import inquirer
import json

from main import handle_benchmark_run, setup_logging
from BenchmarkManager import JobStatus


def _find_interrupted_jobs() -> list:
    """
    Search for results.json files containing interrupted jobs.
    :return: list of directories containing results of interrupted jobs.
    :rtype: None
    """
    possible_dirs = []
    dirs_results = glob.glob("benchmark_runs/**/results.json")
    for dr in dirs_results:
        with open(dr, 'r', encoding='utf-8') as results_json:
            res = json.load(results_json)
            # find all results.json contain interrupted jobs
            if any(r.get("module", {}).get("quark_job_status", JobStatus.UNDEF.name)
                   == JobStatus.INTERRUPTED.name for r in res):
                possible_dirs.append(dr.replace("results.json", ""))
    return possible_dirs


def get_resume_dir(args):
    resume_dir = None
    possible_dirs = _find_interrupted_jobs()
    if len(possible_dirs) == 1:
        resume_dir = possible_dirs[0]
    if len(possible_dirs) > 1:
        answer = inquirer.prompt(
            [inquirer.List("resume_dir",
                           message="Found several interrupted jobs."
                           #    "(You can also specify by --resume-dir <benchmark_run_dir>)\n"
                                   "Please select directory",
                           choices=possible_dirs
                           # add FAILED here if results.json contains info about previous attempts
                           )])
        resume_dir = answer["resume_dir"]
    return resume_dir


def resume(modules, resume_dir):
    args = argparse.Namespace()

    args.modules = modules

    args.resume = True
    args.resume_dir = resume_dir
    args.failfast = False

    args.config = None
    args.createconfig = False

    args.summarize = False

    handle_benchmark_run(args)


def create_resume_parser(parser: argparse.ArgumentParser):
    parser.add_argument('-m', '--modules', help="Provide a file listing the modules to be loaded")
    parser.add_argument('-rd', '--resume-dir', nargs='?', help='Provide results directory of the job to be resumed')


def main():

    # Note for developers: the structure is such that the code could easily be integrated into
    # QUARK main as an additional 'goal' (search for 'goal' in main to understand what I mean).
    parser = argparse.ArgumentParser()
    create_resume_parser(parser)
    args = parser.parse_args()

    setup_logging()

    if args.resume_dir:
        resume_dir = args.resume_dir
    else:
        resume_dir = get_resume_dir(args)
        if not resume_dir:
            logging.info("No interrupted jobs found.")
            exit(0)
    logging.info(resume_dir)

    resume(args.modules, resume_dir)


if __name__ == '__main__':
    main()
