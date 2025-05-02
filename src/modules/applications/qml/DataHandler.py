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

from abc import ABC, abstractmethod

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class DataHandler(ABC):
    """
    Abstract base class for DataHandler. This class defines the
    necessary methods that both supervised and unsupervised QML applciations
    must implement.
    """

    @abstractmethod
    def data_load(self, gen_mod: dict, config: dict) -> tuple[any, float]:
        """
        Helps to ensure that the model can effectively learn the underlying
        patterns and structure of the data, and produce high-quality outputs.

        :param gen_mod: Dictionary with collected information of the previous modules
        :param config: Config specifying the parameters of the data handler
        :return: Mapped problem and the time it took to create the mapping
        """
        pass

    @abstractmethod
    def evaluate(self, solution: any) -> tuple[any, float]:
        """
        Computes the best loss values.

        :param solution: Solution data
        :return: Evaluation data and the time it took to create it
        """
        pass

    @staticmethod
    def tb_to_pd(logdir: str, rep: str) -> None:
        """
        Converts TensorBoard event files in the specified log directory
        into a pandas DataFrame and saves it as a pickle file.

        :param logdir: Path to the log directory containing TensorBoard event files
        :param rep: Repetition counter
        """
        event_acc = EventAccumulator(logdir)
        event_acc.Reload()
        tags = event_acc.Tags()
        data = []
        tag_data = {}
        for tag in tags['scalars']:
            data = event_acc.Scalars(tag)
            tag_values = [d.value for d in data]
            tag_data[tag] = tag_values
        data = pd.DataFrame(tag_data, index=[d.step for d in data])
        data.to_pickle(f"{logdir}/data_{rep}.pkl")
