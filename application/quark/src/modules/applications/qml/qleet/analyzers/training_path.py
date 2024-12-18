"""Module responsible to generating plots of the training trajectory.

The training trajectory is the set of parameter values (projected down to some low
dimensional space) that the model had through the different epochs of it's training
process. This when plotted for one model tells us if the loss was decreasing always,
if the learning rate should be lowered, increased, or what the schedule should look
like, etc. When plotted for more than one model, it let's us know if the paths are
converging or not, giving us a view of how likely is our generated solution optimal.
If many of the models converge to the same path and start mixing, then they likely
are optimal, if not they they are more likely to be just random chance solutions.
"""

import typing as ty

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as pg

from .loss_landscape import LossLandscapePlotter
from ..interface.metas import MetaLogger
from ..simulators.pqc_trainer import PQCSimulatedTrainer


class OptimizationPathPlotter(MetaLogger):
    """Class which logs the parameter information and plots it over the iterations of training.

    This will be used to plot the parameter values in 2-D or 3-D, rather a t-SNE or PCA projection
    or the parameter values. For getting the loss values for the associated training points as a
    part of the plot too, see `LossLandscapePathPlotter`.

    This class conforms to the `MetaLogger` interface and can be used as part of an `AnalyzerList`
    when plotting the training properties of a circuit.
    """

    def __init__(self, mode: str = "tSNE"):
        """Constructs the Path Plotter object.

        :type mode: str
        :param mode: The type of projection we use to show the plots in lower dimensions
        """
        super().__init__()
        assert mode in [
            "tSNE",
            "PCA",
        ], "Mode of Dimensionality Reduction is not implemented, use PCA or tSNE."
        self.dimensionality_reduction = TSNE if mode == "tSNE" else PCA

    def log(self, solver: PQCSimulatedTrainer, _loss: float) -> None:
        """Logs the value of the parameters that the circuit currently has.
        The parameter values should be a numpy vector.

        :type solver: PQCSimulatedTrainer
        :param solver: The trainer module which has the parameters to be plotted
        :type _loss: float
        :param _loss: The loss value at that epoch, not used by this class
        """
        self.data.append(solver.model.trainable_variables[0].numpy())
        self.runs.append(self.trial)
        self.item.append(self.counter)
        self.counter += 1

    def plot(self, large_marker_size=5) -> pg.Figure:
        """Plots the 2D parameter projections.
        For the entire set of runs, the class has logged the parameter values.
        Now it reduces the dimensionality of those parameter vectors using PCA or tSNE
        and then plots them on a 2D plane.

        :returns: The figure on which the parameter projections are plotted
        :rtype: Plotly figure
        """
        raw_params = np.stack(self.data)
        final_params = self.dimensionality_reduction(n_components=2).fit_transform(
            raw_params
        )
        max_number_of_runs = max(self.item)
        size_values = [
            large_marker_size if size > max_number_of_runs - 5 else 1
            for size in self.item
        ]
        fig = px.scatter(
            x=final_params[:, 0],
            y=final_params[:, 1],
            color=self.runs,
            size=size_values,
        )
        return fig


class LossLandscapePathPlotter(MetaLogger):
    """An module to plot the training path of the PQC on the loss landscape

    This class is an extension of the Loss Landscape plotter and the Training
    Path plotter, puts both the ideas together and shows how the different models
    ended us at different parts of the loss landscape.
    """

    def __init__(self, base_plotter: LossLandscapePlotter):
        """Constructor for the LossLandscapePathPlotter.

        :type base_plotter: LossLandscapePlotter
        :param base_plotter: The loss landscape plotter to plot the training path on top of
        """
        super().__init__()
        self.loss: ty.List[float] = []
        self.plotter = base_plotter

    def log(self, solver: "PQCSimulatedTrainer", loss: float):
        """Logs the value of the parameters that the circuit currently has.
        The parameter values should be a numpy vector.

        :type solver: PQCSimulatedTrainer
        :param solver: The trainer module which has the parameters to be plotted
        :type loss: float
        :param loss: The value of the loss at the current epoch
        """
        self.data.append(
            self.plotter.axes @ solver.model.trainable_variables[0].numpy()
        )
        self.loss.append(loss)
        self.runs.append(self.trial)
        self.item.append(self.counter)
        self.counter += 1

    def plot(self):
        """Plots the 2D parameter projections with the loss value on the 3rd dimension.
        For the entire set of runs, the class has logged the parameter values.
        Now it reduces the dimensionality of those parameter vectors using PCA or tSNE
        and then plots them on a 2D plane, associates them with a loss value to put on
        the third dimension. This output is coupled with the actual loss landscape drawing
        and returned.

        :returns: The figure on which the parameter projections are plotted
        :rtype: Plotly figure
        """

        data = np.array(self.data)
        loss = np.array(self.loss)
        max_number_of_runs = max(self.item)
        size_values = np.array(
            [12 if size > max_number_of_runs - 5 else 5 for size in self.item]
        )
        fig = pg.Figure(
            data=[
                pg.Scatter3d(
                    x=data[:, 0],
                    y=data[:, 1],
                    z=-loss,
                    mode="markers",
                    marker=dict(color=self.runs, size=size_values),
                )
            ]
        )
        return fig
