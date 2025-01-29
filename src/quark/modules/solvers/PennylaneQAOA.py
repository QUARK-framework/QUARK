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

import ast
import inspect
import json
import logging
import types
from collections import Counter
from functools import partial, wraps
from time import time
from typing import TypedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from pennylane import numpy as npqml

from quark.modules.solvers.Solver import Solver
from quark.modules.Core import Core
from quark.utils import start_time_measurement, end_time_measurement

matplotlib.use('Agg')


class PennylaneQAOA(Solver):
    """
    Pennylane QAOA solver.
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__()
        self.submodule_options = [
            "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            "arn:aws:braket:::device/quantum-simulator/amazon/tn1",
            "arn:aws:braket:us-east-1::device/qpu/ionq/Harmony",
            "arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3",
            "arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy",
            "braket.local.qubit",
            "default.qubit",
            "default.qubit.autograd",
            "qulacs.simulator",
            "lightning.gpu",
            "lightning.qubit"
        ]

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module.

        :return: List of dict with requirements of this module
        """
        return [
            {"name": "pennylane", "version": "0.39.0"},
            {"name": "pennylane-lightning", "version": "0.39.0"},
            {"name": "amazon-braket-pennylane-plugin", "version": "1.30.2"},
            {"name": "numpy", "version": "1.26.4"}
        ]

    def get_default_submodule(self, option: str) -> Core:
        """
        Returns the default submodule based on the provided option.

        :param option: The name of the submodule
        :return: Instance of the default submodule
        """

        if option == "arn:aws:braket:us-east-1::device/qpu/ionq/Harmony":
            from quark.modules.devices.braket.Ionq import Ionq  # pylint: disable=C0415
            return Ionq("ionq", "arn:aws:braket:us-east-1::device/qpu/ionq/Harmony")
        elif option == "arn:aws:braket:::device/quantum-simulator/amazon/sv1":
            from quark.modules.devices.braket.SV1 import SV1  # pylint: disable=C0415
            return SV1("SV1", "arn:aws:braket:::device/quantum-simulator/amazon/sv1")
        elif option == "arn:aws:braket:::device/quantum-simulator/amazon/tn1":
            from quark.modules.devices.braket.TN1 import TN1  # pylint: disable=C0415
            return TN1("TN1", "arn:aws:braket:::device/quantum-simulator/amazon/tn1")
        elif option == "arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3":
            from quark.modules.devices.braket.Rigetti import Rigetti  # pylint: disable=C0415
            return Rigetti("Rigetti", "arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3")
        elif option == "arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy":
            from quark.modules.devices.braket.OQC import OQC  # pylint: disable=C0415
            return OQC("OQC", "arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy")
        elif option == "braket.local.qubit":
            from quark.modules.devices.HelperClass import HelperClass  # pylint: disable=C0415
            return HelperClass("braket.local.qubit")
        elif option == "default.qubit":
            from quark.modules.devices.HelperClass import HelperClass  # pylint: disable=C0415
            return HelperClass("default.qubit")
        elif option == "default.qubit.autograd":
            from quark.modules.devices.HelperClass import HelperClass  # pylint: disable=C0415
            return HelperClass("default.qubit.autograd")
        elif option == "qulacs.simulator":
            from quark.modules.devices.HelperClass import HelperClass  # pylint: disable=C0415
            return HelperClass("qulacs.simulator")
        elif option == "lightning.gpu":
            from quark.modules.devices.HelperClass import HelperClass  # pylint: disable=C0415
            return HelperClass("lightning.gpu")
        elif option == "lightning.qubit":
            from quark.modules.devices.HelperClass import HelperClass  # pylint: disable=C0415
            return HelperClass("lightning.qubit")
        else:
            raise NotImplementedError(f"Device Option {option} not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this solver.

        :return: Dictionary of configuration settings
        .. code-block:: python

                    return {
                            "shots": {  # number measurements to make on circuit
                                "values": list(range(10, 500, 30)),
                                "description": "How many shots do you need?"
                            },
                            "iterations": {  # number measurements to make on circuit
                                "values": [1, 10, 20, 50, 75],
                                "description": "How many iterations do you need?"
                            },
                            "layers": {
                                "values": [2, 3, 4],
                                "description": "How many layers for QAOA do you want?"
                            },
                            "coeff_scale": {
                                "values": [0.01, 0.1, 1, 10],
                                "description": "How do you want to scale your coefficients?"
                            },
                            "stepsize": {
                                "values": [0.0001, 0.001, 0.01, 0.1, 1],
                                "description": "Which stepsize do you want?"
                            }
                        }

        """
        return {
            "shots": {  # number measurements to make on circuit
                "values": [None] + list(range(10, 500, 30)),
                "description": "How many shots do you need?"
            },
            "iterations": {  # number measurements to make on circuit
                "values": [1, 10, 20, 50, 75],
                "description": "How many iterations do you need?"
            },
            "layers": {
                "values": [1, 2, 3, 4],
                "description": "How many layers for QAOA do you want?"
            },
            "coeff_scale": {
                "values": [0.01, 0.1, 1, 10],
                "description": "How do you want to scale your coefficients?"
            },
            "stepsize": {
                "values": [0.0001, 0.001, 0.01, 0.1, 1],
                "description": "Which stepsize do you want?"
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config.

        .. code-block:: python

            shots: int
            depth: int
            iterations: int
            layers: int
            coeff_scale: float
            stepsize: float

        """
        shots: int
        depth: int
        iterations: int
        layers: int
        coeff_scale: float
        stepsize: float

    @staticmethod
    def normalize_data(data: any, scale: float = 1.0) -> any:
        """
        Not used currently, as I just scale the coefficients in the qaoa_operators_from_ising.

        :param data: Data to normalize
        :param scale: Scaling factor
        :return: Normalized data
        """
        return scale * data / np.max(np.abs(data))

    @staticmethod
    def qaoa_operators_from_ising(j: any, t: any, scale: float = 1.0) -> tuple[any, any]:
        """
        Generates pennylane cost and mixer Hamiltonians from the Ising matrix J and vector t.

        :param j: J matrix
        :param t: t vector
        :param scale: Scaling factor
        :return: Cost Hamiltonian and mixer Hamiltonian
        """
        # Define the scaling factor
        scaling_factor = scale * max(np.max(np.abs(j.flatten())), np.max(np.abs(t)))

        # Scale the coefficients
        j /= scaling_factor
        t /= scaling_factor

        sigzsigz_arr = [
            qml.PauliZ(i) @ qml.PauliZ(k) for i in range(len(j)) for k in range(len(j))
        ]

        sigz_arr = [qml.PauliZ(i) for i in range(len(t))]
        j_real = np.real(j.flatten())
        t_real = np.real(t)
        h_cost = qml.simplify(qml.Hamiltonian([*t_real, *j_real.flatten()], [*sigz_arr, *sigzsigz_arr]))

        # Definition of the mixer hamiltonian
        h_mixer = -1 * qml.qaoa.mixers.x_mixer(range(len(j)))

        return h_cost, h_mixer

    # pylint: disable=R0915
    def run(self, mapped_problem: dict, device_wrapper: any, config: Config, **kwargs: dict) -> tuple[any, float, dict]:
        """
        Runs Pennylane QAOA on the Ising problem.

        :param mapped_problem: Dict containing problem parameters mapped to the Ising model
        :param device_wrapper: Device to run the problem on
        :param config: QAOA solver settings
        :param kwargs: Contains store_dir for the plot of the optimization
        :return: Solution, the time it took to compute it and optional additional information
        """
        j = mapped_problem['J']
        t = mapped_problem['t']
        wires = j.shape[0]
        cost_h, mixer_h = self.qaoa_operators_from_ising(j, t, scale=config['coeff_scale'])

        # set up the problem
        try:
            device_arn = device_wrapper.arn
        except Exception as _:
            device_arn = None

        def qaoa_layer(gamma, alpha):
            qml.qaoa.cost_layer(gamma, cost_h)
            qml.qaoa.mixer_layer(alpha, mixer_h)

        def circuit(params, **kwargs):
            # circuit initialization
            for i in range(wires):
                qml.Hadamard(wires=i)
            # QAOA layers
            qml.layer(qaoa_layer, config['layers'], params[0], params[1])

        # The adjoint differentiation method is preferred over the default 'best' method for the lightning devices
        diff_method = "adjoint" if device_wrapper.device in ["lightning.qubit", "lightning.gpu"] else "best"
        # overwrite default with config
        if "differentiation_method" in config:
            diff_method = config["differentiation_method"]

        # TODO Make this interaction better with aws braket
        if device_arn is None:
            if device_wrapper.device == 'qulacs.simulator':
                dev = qml.device(device_wrapper.device, wires=wires, shots=config['shots'], gpu=True)
            elif device_wrapper.device in ['lightning.qubit', 'lightning.gpu']:
                # No number shots as diff method will be adjoint backprop for these devices
                if diff_method in ["adjoint", "backprop"]:
                    dev = qml.device(device_wrapper.device, wires=wires, shots=None, batch_obs=True)
                else:
                    dev = qml.device(device_wrapper.device, wires=wires, batch_obs=True, shots=config['shots'])
            elif device_wrapper.device == 'default.qubit':
                # No number shots as diff method will be adjoint backprop for these devices
                if diff_method in ["adjoint", "backprop"]:
                    dev = qml.device(device_wrapper.device, shots=None, wires=wires)
                else:
                    dev = qml.device(device_wrapper.device, wires=wires, shots=config['shots'])
            else:
                dev = qml.device(device_wrapper.device, wires=wires, shots=config['shots'])
        else:
            dev = qml.device(
                "braket.aws.qubit",
                device_arn=device_arn,
                wires=wires,
                s3_destination_folder=device_wrapper.s3_destination_folder,
                aws_session=device_wrapper.aws_session,
                parallel=True,
                shots=config['shots'],
                max_parallel=10,
                poll_timeout_seconds=6000,
            )

        @qml.qnode(dev, diff_method=diff_method)
        def cost_function(params):
            circuit(params)
            return qml.expval(cost_h)

        # To measure the QPU execution times we measure the timings of execute and batch_execute
        # in the pennylane devices
        # This is rather experimental and this has to be verified!
        dev.init_array = types.MethodType(monkey_init_array, dev)
        dev.init_array()
        real_decorator = partial(_pseudo_decor, device=dev)
        # TODO At some point validate whether execute and batch_execute are the only and best way to measure quantum
        #  execution times
        # TODO Check the impact of this ast.walk
        # We do this ast walk to assess whether execute is called in batch_execute,
        # which is the case for some simulators. This would distort the timing
        try:
            called_functions = [c.func.attr for c in ast.walk(ast.parse(inspect.getsource(dev.batch_execute).lstrip()))
                                if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)]
        except AttributeError as e:
            logging.warning(f"{e} detected when trying to call pennylane.batch_execute.")
            logging.warning("Instead, tapes will be executed in serial.")
            called_functions = ['execute']
        dev.execute = real_decorator(dev.execute)
        if "execute" not in called_functions:
            dev.batch_execute = real_decorator(dev.batch_execute)

        # Initialize variational parameters randomly
        rand_params = np.random.uniform(size=[2, config['layers']])
        params = npqml.array(rand_params, requires_grad=True)
        logging.info(f"Starting params: {params}")

        # Optimization Loop
        optimizer = qml.GradientDescentOptimizer(stepsize=config['stepsize'])
        logging.info(f"Device: {device_wrapper.device}, Optimizer {optimizer}, Differentiation: {diff_method}, "
                     f"Optimization start...")

        additional_solver_information = {}
        min_param, min_cost, cost_pt, params_list, x = None, None, [], [], []
        run_id = round(time())
        start = start_time_measurement()

        for iteration in range(config['iterations']):
            t0 = start_time_measurement()
            # Evaluates the cost, then does a gradient step to new params
            try:
                params, cost_before = optimizer.step_and_cost(cost_function, params)
            except (ValueError, np.core._exceptions._ArrayMemoryError) as e:  # pylint: disable=W0212
                logging.error(f"When trying to run {optimizer} on {device_wrapper.device}, the following error was "
                              f"encountered")
                logging.error(e)
                logging.error("Run a smaller problem size or select another device.")
                raise e

            # Convert cost_before to a float, so it's easier to handle
            cost_before = float(cost_before)
            if iteration == 0:
                logging.info(f"Initial cost: {cost_before}")
            else:
                logging.info(f"Cost at step {iteration}: {cost_before}")

            # Log the current loss as a metric
            logging.info(f"Time to complete iteration {iteration + 1}: {end_time_measurement(t0)}")
            cost_pt.append(cost_before)
            params_list.append(params)
            x.append(iteration)

            if min_cost is None or min_cost > cost_before:
                min_cost = cost_before
                min_param = params

            if "store_dir" in kwargs:
                plt.figure(figsize=(6, 4))
                plt.plot(x, cost_pt, label='global minimum')
                plt.xlabel("Optimization steps")
                plt.ylabel("Cost / Energy")
                plt.title('/'.join([f"{key}: {value}" for (key, value) in config.items()]))
                plt.legend()
                plt.savefig(f"{kwargs['store_dir']}/plot_pennylane_qaoa_cost_{run_id}.pdf",
                            dpi=300)
                plt.clf()

        params = min_param
        logging.info(f"Final params: {params}")
        logging.info(f"Final costs: {min_cost}")

        @qml.qnode(dev)
        def samples(params):
            circuit(params)
            return [qml.sample(qml.PauliZ(i)) for i in range(wires)]

        def evaluate_params_sampling(params):
            s = samples([params[0], params[1]]).T
            s = (1 - s) / 2
            s = map(tuple, s)
            counts = Counter(s)
            indx = np.ndindex(*[2] * wires)
            probs = {p: counts.get(p, 0) / config['shots'] for p in indx}
            best_bitstring = max(probs, key=probs.get)

            return best_bitstring, probs

        @qml.qnode(dev)
        def probability_circuit(params):
            circuit(params)
            return qml.probs(wires=range(wires))

        def evaluate_params_probs(params):
            probs_raw = np.array(probability_circuit([params[0], params[1]]))
            indx = np.ndindex(*[2] * wires)
            probs = {p: probs_raw[i] for i, p in enumerate(indx)}
            best_bitstring = max(probs, key=probs.get)
            return best_bitstring, probs

        if config['shots'] is None or diff_method in ["backprop", "adjoint"]:
            best_bitstring, probs = evaluate_params_probs(params)
        else:
            best_bitstring, probs = evaluate_params_sampling(params)
        additional_solver_information["quantum_timings"] = dev.timings if len(dev.timings) > 0 else [
            -1]  # hack to make JSON parsing in R work
        additional_solver_information["quantum_timings_sum"] = sum(additional_solver_information["quantum_timings"])
        logging.info(f"{best_bitstring} with {probs[best_bitstring]}")

        logging.info(sorted(probs, key=probs.get, reverse=True)[:5])

        # Save the bitstring with the highest probability per iteration to bitstring_list
        bitstring_list = []
        for el in params_list:
            if config['shots'] is None or diff_method in ["backprop", "adjoint"]:
                bitstring, _ = evaluate_params_probs(el)
            else:
                bitstring, _ = evaluate_params_sampling(el)
            # bitstring, _ = evaluate_params_probs(el) if config['shots'] is None else evaluate_params_sampling(el)
            bitstring_list.append(bitstring)

        # Save the cost, best bitstring, variational parameters per iteration as well as the final prob. distribution
        # TODO: Maybe this can be done more efficient, e.g., only saving the circuit and its weights?
        json_data = {
            'cost': cost_pt,
            'bitstrings': bitstring_list,
            'params': [el.tolist() for el in params_list],  # convert list of tensors to list of lists
            'probs': {str(key): value for key, value in probs.items()}  # convert key (tuples to strings)
        }
        if "store_dir" in kwargs:
            with open(f"{kwargs['store_dir']}/qaoa_details_{run_id}.json", 'w') as fp:
                json.dump(json_data, fp)
        additional_solver_information["run_id"] = run_id
        return best_bitstring, end_time_measurement(start), additional_solver_information


def monkey_init_array(self):
    """
    Here we create the timings array where we later append the quantum timings.
    """
    self.timings = []


def _pseudo_decor(fun, device):
    """
    We use this decorator for measuring execute and batch_execute.
    """

    # Lift the name and doc of the function
    @wraps(fun)
    def ret_fun(*args, **kwargs):
        # Pre function execution here
        from time import time  # pylint: disable=W0621 disable=C0415 disable=W0404
        start_timing = time() * 1000
        returned_value = fun(*args, **kwargs)
        # Post execution here
        device.timings.append(round(time() * 1000 - start_timing, 3))
        return returned_value

    return ret_fun
