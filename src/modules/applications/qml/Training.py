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


class Training(ABC):
    """
    Abstract base class for training QML models.
    """

    @abstractmethod
    def start_training(self, input_data: dict, config: any, **kwargs: dict) -> dict:
        """
        This function starts the training of QML model or deploys a pretrained model.

        :param input_data: A representation of the quantum machine learning model that will be trained
        :param config: Config specifying the parameters of the training (dict-like Config type defined in children)
        :param kwargs: Optional additional settings
        :return: Solution, the time it took to compute it and some optional additional information
        """
        pass
