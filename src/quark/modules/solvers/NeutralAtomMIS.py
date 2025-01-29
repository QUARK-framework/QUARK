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

from typing import TypedDict
import logging

import numpy as np
import pulser

from quark.modules.solvers.Solver import Solver
from quark.modules.Core import Core
from quark.utils import start_time_measurement, end_time_measurement


class NeutralAtomMIS(Solver):
    """
    Neutral atom quantum computer maximum independent sets solver.
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__()
        self.submodule_options = ["MockNeutralAtomDevice"]

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module.

        :return: List of dict with requirements of this module
        """
        return [{"name": "pulser", "version": "1.1.1"}]

    def get_default_submodule(self, option: str) -> Core:
        """
        Returns the default submodule based on the provided option.

        :param option: The name of the submodule
        :return: Instance of the default submodule
        """
        if option == "MockNeutralAtomDevice":
            from quark.modules.devices.pulser.MockNeutralAtomDevice import MockNeutralAtomDevice  # pylint: disable=C0415
            return MockNeutralAtomDevice()
        else:
            raise NotImplementedError(f"Device Option {option} not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this solver.

        :return: Dictionary of configurable settings.
        """
        return {
            "samples": {
                "values": [10, 100, 1000, 10000],
                "custom_input": True,
                "allow_ranges": True,
                "postproc": int,
                "description": "How many samples from the quantum computer do you want per measurement?"
            },
        }

    class Config(TypedDict):
        """
        Attributes of a valid config.

        samples (int): How many times to sample the final state from the quantum computer per measurement
        """
        samples: int

    def run(self, mapped_problem: dict, device_wrapper: any, config: any, **kwargs: dict) -> tuple[list, float, dict]:
        """
        The given application is a problem instance from the pysat library. This uses the rc2 maxsat solver
        given in that library to return a solution.

        :param mapped_problem: Dictionary with graph and register
        :param device_wrapper: Device to run the problem on
        :param config: Solver Configuration
        :param kwargs: Additional settings (not used)
        :return: Solution, the time it took to compute it and optional additional information
        """
        register = mapped_problem.get('register')
        graph = mapped_problem.get('graph')
        nodes = list(graph.nodes())
        edges = list(graph.edges())

        logging.info(f"Got problem with {len(graph.nodes)} nodes, {len(graph.edges)} edges.")

        device = device_wrapper.get_device()
        device.validate_register(register)
        device_backend = device_wrapper.get_backend()
        device_config = device_wrapper.get_backend_config()

        start = start_time_measurement()

        sequence = self._create_sequence(register, device)

        if device_wrapper.device_name == "mock neutral atom device":
            backend = device_backend(sequence, device_config)
            results = backend.run(progress_bar=False)
            sampled_state_counts = results.sample_final_state(N_samples=config['samples'])
        else:
            raise NotImplementedError(f"Device Option {device_wrapper.device_name} not implemented")

        valid_state_counts = self._filter_invalid_states(sampled_state_counts, nodes, edges)
        state = self._select_best_state(valid_state_counts, nodes)
        state_nodes = self._translate_state_to_nodes(state, nodes)

        return state_nodes, end_time_measurement(start), {}

    def _create_sequence(self, register: pulser.Register, device: pulser.devices._device_datacls.Device) \
            -> pulser.Sequence:
        """
        Creates a pulser sequence from a register and a device.

        :param register: The quantum register
        :param device: The device being used
        :return: The created sequence
        """
        pulses = self._create_pulses(device)
        sequence = pulser.Sequence(register, device)
        sequence.declare_channel("Rydberg global", "rydberg_global")
        for pulse in pulses:
            sequence.add(pulse, "Rydberg global")
        return sequence

    def _create_pulses(self, device: pulser.devices._device_datacls.Device) -> list[pulser.Pulse]:
        """
        Creates pulses tuned to MIS problem.

        Pulse creation is a whole art/science on its own that we have not delved into yet.
        If you shape and finetune your pulses to decrease compute time on your neutral atom device.
        We found this configuration in the documentation of the pulser documentation and it works for MIS.
        We are hesitant to make them parametrizable, because setting the wrong values will break your whole MIS.
        Though parameterization of pulses is a feature that we might implement in the future.

        :param device: The device being used
        :return: List of pulses
        """
        omega_max = 2.3 * 2 * np.pi
        delta_factor = 2 * np.pi

        channel = device.channels['rydberg_global']
        max_amp = channel.max_amp
        if max_amp is not None and max_amp < omega_max:
            omega_max = max_amp

        delta_0 = -3 * delta_factor
        delta_f = 1 * delta_factor

        t_rise = 2000
        t_fall = 2000
        t_sweep = (delta_f - delta_0) / (2 * np.pi * 10) * 5000

        rise = pulser.Pulse.ConstantDetuning(
            pulser.waveforms.RampWaveform(t_rise, 0.0, omega_max), delta_0, 0.0
        )
        sweep = pulser.Pulse.ConstantAmplitude(
            omega_max, pulser.waveforms.RampWaveform(t_sweep, delta_0, delta_f), 0.0
        )
        fall = pulser.Pulse.ConstantDetuning(
            pulser.waveforms.RampWaveform(t_fall, omega_max, 0.0), delta_f, 0.0
        )
        pulses = [rise, sweep, fall]

        for pulse in pulses:
            channel.validate_pulse(pulse)

        return pulses

    def _filter_invalid_states(self, state_counts: dict, nodes: list, edges: list) -> dict:
        """
        Filters out invalid states that do not meet the problem constraints.

        :param state_counts: Counts of each sampled data
        :param nodes: List of nodes in the graph
        :param edges: List of edges in the graph
        :return: Dictionary of valid state counts
        """
        valid_state_counts = {}
        for state, count in state_counts.items():
            selected_nodes = self._translate_state_to_nodes(state, nodes)

            is_valid = True
            for edge in edges:
                if edge[0] in selected_nodes and edge[1] in selected_nodes:
                    is_valid = False
                    break
            if is_valid:
                valid_state_counts[state] = count

        return valid_state_counts

    def _translate_state_to_nodes(self, state: str, nodes: list) -> list:
        """
        Translates a state string into the corresponding list of nodes.

        :param state: State string
        :param nodes: List of nodes
        :return: List of nodes corresponding to the states
        """
        return [key for index, key in enumerate(nodes) if state[index] == '1']

    def _select_best_state(self, states: dict, nodes: list) -> str:
        """
        Selects the best state from the available valid states.

        :param states: Dictionary of valid states and their counts
        :param nodes: List of nodes
        :return: The best state as a string
        """
        # TODO: Implement the samplers
        try:
            best_state = max(states, key=lambda k: states[k])
        except Exception:  # pylint: disable=W0702
            # TODO: Specify error
            # TODO: Clean this up
            n_nodes = len(nodes)
            best_state = "0" * n_nodes

        return best_state
