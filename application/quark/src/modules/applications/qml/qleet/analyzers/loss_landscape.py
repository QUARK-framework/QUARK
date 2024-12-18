"""Module to plot the loss landscapes of circuits.

For any variational quantum algorithm being trained to optimize on a given metric,
the plot of a projected subspace of the metric is of value because it helps us
confirm along random axes that our point is indeed the local minima / maxima and
also helps visualize how rough the landscape is giving clues on how likely the
variational models might converge.

We hope that these visualizations can help improve the choice of optimizers and
ansatz we have for these quantum circuits.
"""

import typing as ty

import numpy as np
import tqdm.auto as tqdm
import plotly.graph_objects as pg

from ..simulators.pqc_trainer import PQCSimulatedTrainer
from ..interface.metric_spec import MetricSpecifier
from ..interface.metas import MetaExplorer


class LossLandscapePlotter(MetaExplorer):
    """This class plots the loss landscape for a given PQC trainer object.

    It can plot the true loss that we are training on or on some other metric, this can help
    use proxy metrics as loss functions and seeing if they help optimize on the true target
    metric.

    These plots can support 1-D and 2-D subspace projections for now, since we have to plot
    the loss value on the second or third axis. A 3-D projection of the plot will also be supported
    by v1.0.0 and onwards, which will use colors and point density to show the metric values.
    """

    def __init__(
        self, solver: PQCSimulatedTrainer, metric: MetricSpecifier, dim: int = 2
    ) -> None:
        """Initializes the Loss Landscape plotter.
        The plotter takes a PQC trainer, which will expose the it's present parameters
        and help us sample the outputs of the circuit, be it classical or use the quantum state
        vectors or density matrices, and through those outputs it computes our metric to give
        the 3D contour plot of the metric for perturbations of the parameters near the currently
        trained optima.

        :type solver: PQCSimulatedTrainer
        :param solver: The PQC trainer class, which contains both the
        :type metric: MetricSpecifier
        :param metric: The metric which is being plotted for different parameter values
        :type dim: int
        :param dim: The number of dimensions of the subspace to be sampled,
            necessarily 2 to get a contour plot
        """
        super().__init__()
        self.n = len(solver.circuit.parameters)
        self.metric = metric
        self.solver = solver
        self.dim = dim
        self.axes = self.__random_subspace(dim=self.dim)

    def __random_subspace(self, dim: int) -> np.ndarray:
        """Generates basis vectors for a random subspace
        Performs Gram-Schmidt orthonormalization to generate this set.

        :type dim: int
        :param dim: The number of dimensions the subspace should have
        :returns: The basis set of vectors for our subspace as a 2D numpy matrix

        Note that this only works for Real valued vectors, there are issues with doing this for
        complex vectors to generate unitary matrices, use a different approach for that.
        """
        axes: ty.List[np.ndarray] = []
        for _i in range(dim):
            axis = np.random.random(self.n)
            for other_axis in axes:
                projection = np.dot(axis, other_axis)
                axis = axis - projection * other_axis
            axis = axis / np.linalg.norm(axis)
            axes.append(axis)
        return np.stack(axes, axis=0)

    def scan(
        self, points: int, distance: float, origin: np.ndarray
    ) -> ty.Tuple[np.ndarray, np.ndarray]:
        """Scans the target vector-subspace for values of the metric
        Returns the sampled coordinates in the grid and the values of the metric at those
        coordinates. The sampling of the subspace is done uniformly, and evenly in all directions.

        :type points: int
        :param points: Number of points to sample
        :type distance: float
        :param distance: The range of parameters around the current value to scan over
        :type origin: np.ndarray
        :param origin: The value of the current parameter to be used as origin of our plot
        :returns: tuple of the coordinates and the metric values at those coordinates
        :rtype: a tuple of np.array, shapes being (n, dims) and (n,)
        """
        chained_range = [
            np.linspace(-distance, distance, points) for _i in range(self.dim)
        ]
        coords = np.reshape(
            np.stack(np.meshgrid(*chained_range), axis=-1), (-1, self.dim)
        )
        values = np.zeros(len(coords), dtype=np.float64)
        with tqdm.trange(len(coords)) as iterator:
            iterator.set_description("Contour Plot Scan")
            for i in iterator:
                # TODO: Incorporate state vector and density matrix modes for higher speed
                values[i] = self.metric.from_circuit(
                    circuit_descriptor=self.solver.circuit,
                    parameters=coords[i] @ self.axes + origin,
                    mode="samples",
                )
        return values, coords

    def plot(
        self, mode: str = "surface", points: int = 25, distance: float = np.pi
    ) -> pg.Figure:
        """Plots the loss landscape
        The surface plot is the best 3D visualization, but it uses the plotly dynamic interface,
        it also has an overhead contour. For simple 2D plots which can be used as matplotlib
        graphics or easily used in publications, use line and contour modes.

        :type mode: str
        :param mode: line, contour or surface, what type of plot do we want?
        :type points: int
        :param points: number of points to sample for the metric
        :type distance: float
        :param distance: the range around the current parameters that we need to sample to
        :returns: The figure object that has been generated
        :rtype: Plotly or matplotlib figure object
        :raises NotImplementedError: For the 1D plotting. TODO Implement 1D plots.

        Increasing the number of points improves the quality of the plot but takes a
        lot more time, it scales quadratically in the number of points. Lowering the
        distance is a good idea if using fewer points, since you get the same number
        of points for a small region. Note that these plots can be deceptive, there might
        be large ridges that get missed due to lack of resolution of the points,
        always be careful and try to use as many points as possible before making
        a final inference.
        """
        assert mode in ["line", "contour", "surface"]
        if mode == "contour":
            assert (
                self.dim == 2
            ), "Contour plots can only be drawn with 2-dimensional axes"
            origin = self.solver.model.trainable_variables[0]
            data, _coords = self.scan(points, distance, origin)
            data = np.reshape(data, (points, points))
            scan_range = np.linspace(-distance, +distance, points)
            fig = pg.Figure(data=pg.Contour(z=data, x=scan_range, y=scan_range))
            return fig
        elif mode == "surface":
            assert (
                self.dim == 2
            ), "Contour plots can only be drawn with 2-dimensional axes"
            origin = self.solver.model.trainable_variables[0]
            data, _coords = self.scan(points, distance, origin)
            data = np.reshape(data, (points, points))
            scan_range = np.linspace(-distance, +distance, points)
            fig = pg.Figure(data=pg.Surface(z=data, x=scan_range, y=scan_range))
            return fig
        else:
            raise NotImplementedError("This plotting mode has not been implemented yet")
