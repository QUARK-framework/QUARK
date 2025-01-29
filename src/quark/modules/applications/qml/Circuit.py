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
from quark.modules.Core import Core


class Circuit(Core, ABC):
    """
    This module is abstract base class for the library-agnostic gate sequence, that define a quantum circuit.
    """

    @abstractmethod
    def generate_gate_sequence(self, input_data: dict, config: any) -> dict:
        """
        Generates the library agnostic gate sequence, a well-defined definition of the quantum circuit.

        :param input_data: Input data required to generate the gate sequence
        :param config: Configuration for the gate sequence
        :return: Generated gate sequence
        """
        pass
