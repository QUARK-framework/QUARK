"""Module responsible to generating histogram plots of the parameter values.

This modules is our alternative for TensorBoard histograms. It shows the 3D histograms for
one parameter over an ensemble of models or that for a group of parameters. As the models trains
those histograms should change from the initial distribution they were initialized to, to the
final optimal distribution they converge to. The parameter which don't really change their
distribution in training may be either not contributing to the model or are not being trained
well.
"""

import typing
import numpy as np

import sympy
from matplotlib import pyplot as plt
import seaborn as sns

from ..interface.metas import MetaExplorer
from ..interface.circuit import CircuitDescriptor
from ..simulators.pqc_trainer import PQCSimulatedTrainer


class ParameterHistograms(MetaExplorer):
    """Class to plot the histograms of parameters in the circuit."""

    def __init__(
        self,
        circuit: CircuitDescriptor,
        ensemble_size: int = 3,
        groups: typing.Optional[typing.Dict[str, typing.List[sympy.Symbol]]] = None,
        epochs_chart=(0, 10, 10),
    ) -> None:
        """Creates an explorer object which will plot the histogram.
        We specify which parameter will be grouped together to be plotted in the same histogram,
        as well as after how many epochs do we want to draw the plots.

        :type circuit: CircuitDescriptor
        :param circuit: The Parametrized Quantum circuit
        :type grous: Dict mapping strings to a list of `sympy.Symbol`s
        :param groups: Groups of variables which can be analyzed together.
        :type ensemble_size: int
        :param ensemble_size: The number of models in the ensemble
        :type epochs_chart: tuple of int
        :param epochs_chart: The list of number of epochs in each block

        The epochs chart presents the number of iterations of training in each block, after each
        block of all the models we shall plot the histograms of the parameters, so the plotting
        is not done after every single epochs and the spacing is left customizable to the user.

        The parameter groups have associated group names which are the keys of the dictionary, we
        use them to label the plots.

        TODO: Get the trainable model class as an input, convert this to a logger
        """
        super().__init__()
        self.circuit = circuit
        # Generate an ensemble or runs
        self.ensemble_size = ensemble_size
        self.models = [
            PQCSimulatedTrainer(self.circuit) for _ in range(self.ensemble_size)
        ]
        self.epochs_chart = epochs_chart
        # Prepare the groups of variables which will be analyzed together
        if groups is not None:
            self.groups = groups
        else:
            self.groups = dict()
            for param in circuit.parameters:
                self.groups[param.name] = [param]
        # Prepare the array to store histograms resulting from simulation
        self._histograms: typing.Dict[str, typing.List[typing.List]] = {
            group: [[] for _ in self.epochs_chart] for group in self.groups.keys()
        }

    def simulate(self) -> None:
        """Simulates the circuit and generate the histogram data.

        This is training an ensemble of models for the same number of epochs,
        which is extracted from the epochs chart property. After each block of training
        of all the models, the parameter are extracted and stored to be plotted later.
        """
        for epochs_idx, epochs_to_train in enumerate(self.epochs_chart):
            for model in self.models:
                model.train(n_samples=epochs_to_train)
            for group_name, group_symbols in self.groups.items():
                for model in self.models:
                    for variable in group_symbols:
                        value = self._get_symbol_value_from_model(model, variable)
                        self._histograms[group_name][epochs_idx].append(value)
                        print(f"{variable} in {epochs_idx}: {value}")

    @staticmethod
    def _get_symbol_value_from_model(
        model: PQCSimulatedTrainer, symbol: sympy.Symbol
    ) -> float:
        """Get the current value of the symbol in the PQC Trainer

        :type model: PQCSimulatedTrainer
        :param model: The model we want to find the symbol values from
        :type sybmol: sympy.Symbol
        :param symbol: The sympy symbol we want to find the value of
        :return: The current value of the symbol
        :rtype: float
        """
        return model.pqc_layer.symbol_values()[symbol]

    def plot(self) -> np.ndarray:
        """Plot the parameter histogram for this circuit.
        The plots are layed out with the different epochs of training along one axis and
        the different parameter groups on the other. All the values of the same parameter
        group for the same epoch block over all the models are in the ensemble are plotted
        in a single histogram.

        :return: The axes with the completed plots
        :rtype: plt.Axes

        Try to ensure that a substantial number of parameters are part of each histogram,
        so use large groups or train more models in the ensemble to make the plots meaningful.
        """
        self.simulate()
        _fig, ax = plt.subplots(
            len(self.epochs_chart),
            len(self.groups),
            figsize=(len(self.groups) * 5, len(self.epochs_chart) * 5),
        )
        for group_idx, group_name in enumerate(self.groups.keys()):
            for epoch_idx in range(len(self.epochs_chart)):
                sns.kdeplot(
                    self._histograms[group_name][epoch_idx],
                    ax=ax[epoch_idx, group_idx],
                )
                ax[epoch_idx, group_idx].set_title(f"{group_name} @ epoch:{epoch_idx}")
                ax[epoch_idx, group_idx].set_xlabel("Parameter Values")
        return ax
