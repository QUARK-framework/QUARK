"""Module to format the results.json to old schema for validation and
for dash viewer"""

#  Copyright 2023 science + computing ag
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#

import logging
import pandas as pd


def flatten_data_field(data_frame: pd.DataFrame, field_name, json_field_path: list[str]):
    """A safe method to format the dataframe, while skipping non-existing fields and proceed"""
    def trafo(row):
        val = row
        debugging_info = "results"
        for path_step in json_field_path:
            try:
                val = val[path_step]
                debugging_info = f"{debugging_info}.{path_step}"
            except KeyError as k:
                raise KeyError(
                    f"No element called '{path_step}' in {debugging_info}") from k

        return val

    try:
        data_frame[field_name] = data_frame.apply(trafo, axis=1)

    except KeyError as k:
        logging.error(k)
        data_frame[field_name] = None


def jsonstyle_to_csvstyle(data_frame: pd.DataFrame):
    """extract the datafields from the json-based dataframe to 
    match the csv-based ones appearing in QUARK 2.0"""

    flatten_data_field(data_frame, "application", ["module", "module_name"])
    flatten_data_field(data_frame, "application_config",
                       ["module", "module_config"])
    flatten_data_field(data_frame, 'solution_validity',
                       ["module", "solution_validity"])
    flatten_data_field(data_frame, 'solution_quality',
                       ["module", "solution_quality"])
    flatten_data_field(data_frame, 'time_to_solve', ["total_time"])
    flatten_data_field(data_frame, 'time_to_solve_unit', ["total_time_unit"])
    flatten_data_field(data_frame, 'mapping_config', [
                       "module", "submodule", "module_config"])
    flatten_data_field(data_frame, 'mapping', [
                       "module", "submodule", "module_name"])
    flatten_data_field(data_frame, 'solver_config', [
                       "module", "submodule", "submodule", "module_config"])
    flatten_data_field(data_frame, 'solver', [
                       "module", "submodule", "submodule", "module_name"])
    flatten_data_field(data_frame, 'solution_raw', [
                       "module", "submodule", "submodule", "solution_raw"])
    flatten_data_field(data_frame, 'additional_solver_information', [
                       "module", "submodule", "submodule", "additional_solver_information"])
    flatten_data_field(data_frame, 'device_config', [
                       "module", "submodule", "submodule", "submodule", "module_config"])
    flatten_data_field(data_frame, 'device', [
                       "module", "submodule", "submodule", "submodule", "module_name"])
    flatten_data_field(data_frame, 'solution_quality_unit',
                       ["module", "solution_quality_unit"])
    flatten_data_field(data_frame, 'time_to_evaluation', ["total_time"])
    flatten_data_field(data_frame, 'time_to_evaluation_unit', [
                       "total_time_unit"])
    flatten_data_field(data_frame, 'time_to_mapping', ['module', "total_time"])
    flatten_data_field(data_frame, 'time_to_mapping_unit',
                       ['module', "total_time_unit"])
    flatten_data_field(data_frame, 'time_to_process_solution', [
                       'module', 'submodule', "total_time"])
    flatten_data_field(data_frame, 'time_to_process_solution_unit', [
                       'module', 'submodule', "total_time_unit"])
    flatten_data_field(data_frame, 'time_to_reverse_map',
                       ['module', "postprocessing_time"])
    flatten_data_field(data_frame, 'time_to_reverse_map_unit', [
                       'module', "postprocessing_time_unit"])
    flatten_data_field(data_frame, 'time_to_solution', [
                       'module', 'submodule', 'submodule', "total_time"])
    flatten_data_field(data_frame, 'time_to_solution_unit', [
                       'module', 'submodule', 'submodule', "total_time_unit"])
