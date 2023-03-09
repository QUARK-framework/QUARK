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

from typing import final


class Metrics:
    """
    Metrics Module, used by every QUARK module
    """

    def __init__(self, module_name: str, module_src: str):
        """
        Constructor method
        """
        self.module_name = module_name
        self.module_src = module_src
        self.preprocessing_time = None
        self.preprocessing_time_unit = "ms"
        self.postprocessing_time = None
        self.module_config = None
        self.total_time = None
        self.total_time_unit = "ms"
        self.postprocessing_time_unit = "ms"
        self.additional_metrics = {}

    @final
    def validate(self):
        """
        Validates whether the mandatory metrics got recorded, then sets total time.

        :rtype: None
        """
        assert self.preprocessing_time is not None, "preprocessing time must not be None!"
        assert self.postprocessing_time is not None, "postprocessing time must not be None!"
        self.total_time = self.preprocessing_time + self.postprocessing_time

    @final
    def set_preprocessing_time(self, value: float):
        self.preprocessing_time = value

    @final
    def set_module_config(self, config: dict):
        self.module_config = config

    @final
    def set_postprocessing_time(self, value: float):
        self.postprocessing_time = value

    @final
    def add_metric(self, name: str, value: any):
        """
        Method to add a single metric.

        :param name: name of the metric
        :type name: str
        :param value: value of the metric
        :type value: any
        :return:
        :rtype: None
        """
        self.additional_metrics.update({name: value})

    @final
    def add_metric_batch(self, key_values: dict):
        """
        Adds a dictionary containing metrics to the existing metrics.

        :param key_values: dict containing metrics
        :type key_values: dict
        :return:
        :rtype: None
        """
        self.additional_metrics.update(key_values)

    @final
    def reset(self):
        """
        Resets all recorded metrics

        :rtype: None
        """
        self.preprocessing_time = None
        self.postprocessing_time = None
        self.additional_metrics = {}

    @final
    def get(self) -> dict:
        """
        Returns all recorded metrics.

        :return: metrics as a dict
        :rtype: dict
        """
        return {
            "module_name": self.module_name,
            "module_src": self.module_src,
            "module_config": self.module_config,
            "total_time": self.total_time,
            "total_time_unit": self.total_time_unit,
            "preprocessing_time": self.preprocessing_time,
            "preprocessing_time_unit": self.preprocessing_time_unit,
            "postprocessing_time": self.postprocessing_time,
            "postprocessing_time_unit": self.postprocessing_time_unit,
            **self.additional_metrics
        }
