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
import inspect
import ast
from collections import Counter
from typing import TypedDict, Union
import types
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from pennylane import numpy as npqml
import json
from functools import partial, wraps

from devices.braket.Ionq import Ionq
from devices.braket.Rigetti import Rigetti
from devices.braket.OQC import OQC
from devices.braket.SV1 import SV1
from devices.braket.TN1 import TN1
from devices.HelperClass import HelperClass
from solvers.Solver import *


class PennylaneQAOA(Solver):
    """
    Pennylane QAOA solver.
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.device_options = ["arn:aws:braket:::device/quantum-simulator/amazon/sv1",
                               "arn:aws:braket:::device/quantum-simulator/amazon/tn1",
                               "arn:aws:braket:::device/qpu/ionq/ionQdevice",
                               "arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-2",
                               "arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy",
                               "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
                               "braket.local.qubit",
                               "default.qubit",
                               "default.qubit.autograd",
                               "qulacs.simulator",
                               "lightning.gpu",
                               "lightning.qubit"]

    def get_device(self, device_option: str) -> Union[Ionq, SV1, TN1, Rigetti, OQC, HelperClass]:
        if device_option == "arn:aws:braket:::device/qpu/ionq/ionQdevice":
            return Ionq("ionq", "arn:aws:braket:::device/qpu/ionq/ionQdevice")
        elif device_option == "arn:aws:braket:::device/quantum-simulator/amazon/sv1":
            return SV1("SV1", "arn:aws:braket:::device/quantum-simulator/amazon/sv1")
        elif device_option == "arn:aws:braket:::device/quantum-simulator/amazon/tn1":
            return TN1("TN1", "arn:aws:braket:::device/quantum-simulator/amazon/tn1")
        elif device_option == "arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-2":
            return Rigetti("Rigetti", "arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-2")
        elif device_option == "arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy":
            return OQC("OQC", "arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy")
        elif device_option == "braket.local.qubit":
            return HelperClass("braket.local.qubit")
        elif device_option == "default.qubit":
            return HelperClass("default.qubit")
        elif device_option == "default.qubit.autograd":
            return HelperClass("default.qubit.autograd")
        elif device_option == "qulacs.simulator":
            return HelperClass("qulacs.simulator")
        elif device_option == "lightning.gpu":
            return HelperClass("lightning.gpu")
        elif device_option == "lightning.qubit":
            return HelperClass("lightning.qubit")
        else:
            raise NotImplementedError(f"Device Option {device_option} not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this solver

        :return:
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
        Attributes of a valid config

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

        :param data:
        :type data: any
        :param scale:
        :type scale: float
        :return: Normalized data
        :rtype: any
        """
        return scale * data / np.max(np.abs(data))

    @staticmethod
    def qaoa_operators_from_ising(J: any, t: any, scale: float = 1.0) -> (any, any):
        """
        Generates pennylane cost and mixer hamiltonians from the Ising matrix J and vector t.

        :param J: J matrix
        :type J: any
        :param t: t vector
        :type t: any
        :param scale:
        :type scale: float
        :return:
        :rtype: tuple(any, any)
        """
        # we define the scaling factor as scale * the maximum parameter found in the coefficients
        scaling_factor = scale * max(np.max(np.abs(J.flatten())), np.max(np.abs(t)))
        # we scale the coefficients
        J /= scaling_factor
        t /= scaling_factor

        sigzsigz_arr = np.array(
            [[qml.PauliZ(i) @ qml.PauliZ(j) for i in range(len(J))]
             for j in range(len(J))])

        sigz_arr = [qml.PauliZ(i) for i in range(len(t))]
        # one body terms (h_i * sig_z^(i))
        # two body terms (J_ij * sig_z^(i) \otimes * sig_z^(j))
        # total cost function
        h_cost = qml.Hamiltonian([*t, *J.flatten()], [*sigz_arr, *sigzsigz_arr.flatten()],
                                 simplify=True)

        # definition of the mixer hamiltonian
        h_mixer = -1 * qml.qaoa.mixers.x_mixer(range(len(J)))

        return h_cost, h_mixer

    def run(self, mapped_problem: any, device_wrapper: any, config: Config, **kwargs: dict) -> (any, any, float):
        """
        Runs Pennylane QAOA on the Ising problem.

        :param mapped_problem: Ising
        :type mapped_problem: any
        :param device_wrapper:
        :type device_wrapper: any
        :param config:
        :type config: Config
        :param kwargs: contains store_dir for the plot of the optimization
        :type kwargs: any
        :return: Solution, the time it took to compute it and optional additional information
        :rtype: tuple(list, float, dict)
        """

        J = mapped_problem['J']
        t = mapped_problem['t']
        wires = J.shape[0]
        cost_h, mixer_h = self.qaoa_operators_from_ising(J, t, scale=config['coeff_scale'])

        # set up the problem
        try:
            device_arn = device_wrapper.arn
        except Exception:
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

        # TODO Make this interaction better with aws braket
        if device_arn is None:
            if device_wrapper.device == 'qulacs.simulator':
                dev = qml.device(device_wrapper.device, wires=wires, shots=config['shots'], gpu=True)
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
                max_parallel=20,
                poll_timeout_seconds=30,
            )

        # The adjoint differentiation method is preferred over the default 'best' method for the lightning devices
        diff_method = "adjoint" if device_wrapper.device == "lightning.qubit" or device_wrapper.device == "lightning.gpu" else "best"

        @qml.qnode(dev, diff_method=diff_method)
        def cost_function(params):
            circuit(params)
            return qml.expval(cost_h)

        # To measure the QPU execution times we measure the timings of execute and batch_execute in the pennylane devices
        # This is rather experimental and this has to be verified!
        dev.init_array = types.MethodType(monkey_init_array, dev)
        dev.init_array()
        real_decorator = partial(_pseudo_decor, device=dev)
        # TODO At some point validate whether execute and batch_execute are the only and best way to measure quantum
        #  execution times
        # TODO Check the impact of this ast.walk
        # We do this ast walk to assess whether execute is called in batch_execute, which is the case for some simulators.
        # This would distort the timing
        called_functions = [c.func.attr for c in ast.walk(ast.parse(inspect.getsource(dev.batch_execute).lstrip()))
                            if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)]
        dev.execute = real_decorator(dev.execute)
        if "execute" not in called_functions:
            dev.batch_execute = real_decorator(dev.batch_execute)

        # Initialize variational parameters randomly
        rand_params = np.random.uniform(size=[2, config['layers']])
        params = npqml.array(rand_params, requires_grad=True)
        logging.info(f"Starting params: {params}")

        # Optimization Loop
        # optimizer = qml.GradientDescentOptimizer(stepsize=config['stepsize'])
        optimizer = qml.MomentumOptimizer(stepsize=config['stepsize'], momentum=0.9)
        logging.info(f"Optimization start")

        additional_solver_information = {}
        min_param = None
        min_cost = None
        cost_pt = []
        params_list = []
        x = []
        run_id = round(time())
        start = time() * 1000
        for iteration in range(config['iterations']):
            t0 = time()
            # Evaluates the cost, then does a gradient step to new params
            params, cost_before = optimizer.step_and_cost(cost_function, params)
            # Convert cost_before to a float, so it's easier to handle
            cost_before = float(cost_before)
            t1 = time()
            if iteration == 0:
                logging.info(f"Initial cost: {cost_before}")
            else:
                logging.info(f"Cost at step {iteration}: {cost_before}")
            # Log the current loss as a metric
            logging.info(f"Time to complete iteration {iteration + 1}: {t1 - t0} seconds")
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
                plt.title('/'.join(['%s: %s' % (key, value) for (key, value) in config.items()]))
                plt.legend()
                plt.savefig(f"{kwargs['store_dir']}/plot_pennylane_qaoa_cost_{run_id}_{kwargs['repetition']}.pdf",
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

        best_bitstring, probs = evaluate_params_probs(params) if config['shots'] is None else evaluate_params_sampling(
            params)
        additional_solver_information["quantum_timings"] = dev.timings
        additional_solver_information["quantum_timings_sum"] = sum(additional_solver_information["quantum_timings"])
        logging.info(f"{best_bitstring} with {probs[best_bitstring]}")

        logging.info(sorted(probs, key=probs.get, reverse=True)[:5])

        # Save the bitstring with the highest probability per iteration to bitstring_list
        bitstring_list = []
        for el in params_list:
            bitstring, _ = evaluate_params_probs(el) if config['shots'] is None else evaluate_params_sampling(el)
            bitstring_list.append(bitstring)

        # Save the cost, best bitstring, variational parameters per iteration as well as the final prob. distribution
        # TODO: Maybe this can be done more efficient, e.g. only saving the circuit and its weights?
        json_data = {
            'cost': cost_pt,
            'bitstrings': bitstring_list,
            'params': [el.tolist() for el in params_list],  # convert list of tensors to list of lists
            'probs': {str(key): value for key, value in probs.items()}  # convert key (tuples to strings)
        }
        if "store_dir" in kwargs:
            with open(f"{kwargs['store_dir']}/qaoa_details_{run_id}_{kwargs['repetition']}.json", 'w') as fp:
                json.dump(json_data, fp)
        additional_solver_information["run_id"] = run_id
        return best_bitstring, round(time() * 1000 - start, 3), additional_solver_information


def monkey_init_array(self):
    """
    Here we create the timings array where we later append the quantum timings
    :param self:
    :return:
    """
    self.timings = []


def _pseudo_decor(fun, device):
    """
    Massive shoutout to this guy: https://stackoverflow.com/a/25827070/10456906
    We use this decorator for measuring execute and batch_execute
    """

    # magic sauce to lift the name and doc of the function
    @wraps(fun)
    def ret_fun(*args, **kwargs):
        # pre function execution stuff here
        from time import time
        start_timing = time() * 1000
        returned_value = fun(*args, **kwargs)
        # post execution stuff here
        device.timings.append(round(time() * 1000 - start_timing, 3))
        return returned_value

    return ret_fun
