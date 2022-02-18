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

from collections import Counter
from typing import TypedDict, Union
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
# from pennylane import numpy as np

from devices.braket.Ionq import Ionq
from devices.braket.LocalSimulator import LocalSimulator
from devices.braket.Rigetti import Rigetti
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
                               "arn:aws:braket:::device/qpu/rigetti/Aspen-9",
                               "braket.local.qubit",
                               "default.qubit",
                               "default.qubit.autograd",
                               "qulacs.simulator"]

    def get_device(self, device_option: str) -> Union[Ionq, SV1, TN1, Rigetti, HelperClass]:
        if device_option == "arn:aws:braket:::device/qpu/ionq/ionQdevice":
            return Ionq("ionq", "arn:aws:braket:::device/qpu/ionq/ionQdevice")
        elif device_option == "arn:aws:braket:::device/quantum-simulator/amazon/sv1":
            return SV1("SV1", "arn:aws:braket:::device/quantum-simulator/amazon/sv1")
        elif device_option == "arn:aws:braket:::device/quantum-simulator/amazon/tn1":
            return TN1("TN1", "arn:aws:braket:::device/quantum-simulator/amazon/tn1")
        elif device_option == "arn:aws:braket:::device/qpu/rigetti/Aspen-9":
            return Rigetti("Rigetti", "arn:aws:braket:::device/qpu/rigetti/Aspen-9")
        elif device_option == "braket.local.qubit":
            return HelperClass("braket.local.qubit")
        elif device_option == "default.qubit":
            return HelperClass("default.qubit")
        elif device_option == "default.qubit.autograd":
            return HelperClass("default.qubit.autograd")
        elif device_option == "qulacs.simulator":
            return HelperClass("qulacs.simulator")
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
                                            "values": [10, 20, 50, 75],
                                            "description": "How many iterations do you need?"
                                        },
                                        "layers": {
                                            "values": [2, 3, 4],
                                            "description": "How many layers for QAOA do you want?"
                                        },
                                        "coeff_scale": {
                                            "values": [0.01, 0.1, 1, 10],
                                            "description": "How do you want to scale your coefficents?"
                                        },
                                        "stepsize": {
                                            "values": [0.0001, 0.001, 0.01, 0.1, 1],
                                            "description": "Which stepsize do you want?"
                                        }
                                    }

        """
        return {
            "shots": {  # number measurements to make on circuit
                "values": list(range(10, 500, 30)),
                "description": "How many shots do you need?"
            },
            "iterations": {  # number measurements to make on circuit
                "values": [10, 20, 50, 75],
                "description": "How many iterations do you need?"
            },
            "layers": {
                "values": [2, 3, 4],
                "description": "How many layers for QAOA do you want?"
            },
            "coeff_scale": {
                "values": [0.01, 0.1, 1, 10],
                "description": "How do you want to scale your coefficents?"
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

        :param J: matrix J
        :type J: any
        :param t: vector t
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

        # We also need to generate the mixer hamiltonian, implemented in the qml library
        h_mixer = qml.qaoa.mixers.x_mixer(range(len(J)))

        return h_cost, h_mixer

    def run(self, mapped_problem: any, device_wrapper: any, config: Config, **kwargs: dict) -> (any, float):
        """
        Runs Pennylane QAOA on the Ising problem.

        :param mapped_problem: dictionary with the keys 'J' and 't'
        :type mapped_problem: any
        :param device_wrapper:
        :type device_wrapper: any
        :param config:
        :type config: Config
        :param kwargs: contains store_dir for the plot of the optimization
        :type kwargs: any
        :return: Solution and the time it took to compute it
        :rtype: tuple(any, float)
        """

        J = mapped_problem['J']
        t = mapped_problem['t']
        wires = J.shape[0]
        cost_h, mixer_h = self.qaoa_operators_from_ising(J, t, scale=config['coeff_scale'])

        # set up the problem
        try:
            device_arn = device_wrapper.arn
        except Exception as e:
            device_arn = None

        def qaoa_layer(gamma, alpha):
            qml.qaoa.cost_layer(gamma, cost_h)
            qml.qaoa.mixer_layer(alpha, mixer_h)

        def circuit(params, **kwargs):
            for i in range(wires):
                qml.Hadamard(wires=i)
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

        cost_function = qml.ExpvalCost(circuit, cost_h, dev, optimize=True)
        rand_params = np.random.uniform(size=[2, config['layers']])
        params = rand_params  # np.array(rand_params, requires_grad=True)
        logging.info(f"Starting params: {params}")

        # Optimization Loop
        # optimizer = qml.GradientDescentOptimizer(stepsize=config['stepsize'])
        optimizer = qml.MomentumOptimizer(stepsize=config['stepsize'], momentum=0.9)
        logging.info(f"Optimization start")

        min_param = None
        min_cost = None
        cost_pt = []
        x = []
        run_id = round(time())
        start = time() * 1000

        for iteration in range(config['iterations']):
            t0 = time()
            # Evaluates the cost, then does a gradient step to new params
            params, cost_before = optimizer.step_and_cost(cost_function, params)
            # Convert cost_before to a float so it's easier to handle
            cost_before = float(cost_before)
            t1 = time()
            if iteration == 0:
                logging.info(f"Initial cost: {cost_before}")
            else:
                logging.info(f"Cost at step {iteration}: {cost_before}")
            # Log the current loss as a metric
            logging.info(f"Time to complete iteration {iteration + 1}: {t1 - t0} seconds")

            cost_pt.append(cost_before)
            x.append(iteration)

            if min_cost is None or min_cost > cost_before:
                min_cost = cost_before
                min_param = params

            final_cost = float(cost_function(params))
            if "store_dir" in kwargs:
                fig = plt.figure(figsize=(6, 4))
                plt.plot(x, cost_pt, label='global minimum')
                plt.xlabel("Optimization steps")
                plt.ylabel("Cost / Energy")
                plt.title('/'.join(['%s: %s' % (key, value) for (key, value) in config.items()]))
                plt.legend()
                plt.savefig(f"{kwargs['store_dir']}/plot_pennylane_qaoa_cost_{run_id}_{kwargs['repetition']}.pdf",
                            dpi=300)

        params = min_param

        logging.info(f"Final params: {params}")
        logging.info(f"Final costs: {min_cost}")

        # sample measured bitstrings 10000 times
        shots = 10000

        @qml.qnode(dev)
        def samples(params):
            circuit(params)
            return [qml.sample(qml.PauliZ(i)) for i in range(wires)]

        s = samples([params[0], params[1]]).T
        s = (1 - s) / 2
        s = map(tuple, s)

        counts = Counter(s)
        indx = np.ndindex(*[2] * wires)
        probs = {p: counts.get(p, 0) / shots for p in indx}
        best_bitstring = max(probs, key=probs.get)
        logging.info(f"{best_bitstring} with {probs[best_bitstring]}")

        logging.info(sorted(probs, key=probs.get, reverse=True)[:5])

        return best_bitstring, round(time() * 1000 - start, 3)
