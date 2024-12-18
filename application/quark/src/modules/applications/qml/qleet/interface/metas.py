"""This module houses the interfaces for analyzers, and provide a utility container AnalyzerList

* MetaLogger is interface for those analyzers which need the state of the
    circuit at each timestep in training.
* MetaExplorer is the interaface for those analyzers which can generate
    properties from a single snapshot of the circuit.
* AnalyzerList is the convinience container which acts as a list of analyzers
    which easy to use API.
"""

import typing
from abc import abstractmethod, ABC

if typing.TYPE_CHECKING:
    from ..simulators.pqc_trainer import PQCSimulatedTrainer


class MetaLogger(ABC):
    """Abstract class to represent interface of logging.
    Logs the present state of the model during training.
    """

    def __init__(self):
        """Constructs the Logger object."""
        self.trial, self.counter = 0, 0
        self.data = []
        self.runs = []
        self.item = []

    @abstractmethod
    def log(self, solver: "PQCSimulatedTrainer", loss: float):
        """Logs information at one timestep about either the solver or the present loss.

        :type solver: PQCSimulatedTrainer
        :param solver: The state of the PQC trainer at the current timestep
        :type loss: float
        :param loss: The loss at the current timestep
        """
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        """Plots the values logged by the logger."""
        raise NotImplementedError

    def next(self):
        """Moves the logger to analyzing the next run or model path."""
        self.trial += 1
        self.counter = 0


class MetaExplorer(ABC):
    """Abstract class to represent interface of analyzing a the current state of the circuit.
    Treats the parameters of the circuit as a snapshot.
    """

    def __init__(self):
        """Constructs the Explorer object."""


class AnalyzerList:
    """Container class, Stores a list of loggers.

    All the loggers can be asked to log the information they need together.
    The information to be logged can be provided to the Analyzer List in one convinient
    function call, and all the associated functions for all the loggers get called
    which can accept that form of data. All the loggers can also together be moved to
    the next model.
    """

    def __init__(self, *args: typing.Union[MetaLogger, MetaExplorer]):
        """Constructor for the Analyzer List
        Takes the list of MetaLoggers and MetaExplorers as input.
        """
        self._analyzers: typing.Tuple[
            typing.Union[MetaLogger, MetaExplorer]
        ] = typing.cast(typing.Tuple[typing.Union[MetaLogger, MetaExplorer]], args)

    def __str__(self) -> str:
        return "\n".join([str(analyzer) for analyzer in self._analyzers])

    def log(self, solver: "PQCSimulatedTrainer", loss: float) -> None:
        """Logs the current state of model in all the loggers.
        Does not ask the `MetaAnalyzers` to log the information since they don't
        implement the logging interface.
        :type solver: PQCSimulatedTrainer
        :param solver: The PQC trainer whose parameters are to be logged
        :type loss: float
        :param loss: Loss value on the current epoch
        """
        for analyzer in self._analyzers:
            if isinstance(analyzer, MetaLogger):
                analyzer.log(solver, loss)

    def next(self) -> None:
        """Moves the loggers to logging of the next model.
        Completes the logging for the current training path."""
        for analyzer in self._analyzers:
            if isinstance(analyzer, MetaLogger):
                analyzer.next()

    def __getitem__(self, item: int) -> typing.Union[MetaLogger, MetaExplorer]:
        """Returns the given Logger or Explorer
        :type item: int
        :param item: The index of the analyzer we want
        :returns: The analyzer at the given position
        :rtype: `MetaLogger` or `MetaExplorer`
        """
        return self._analyzers[item]

    def __iter__(self) -> typing.Iterable[typing.Union[MetaLogger, MetaExplorer]]:
        """Allows iteration over all the Loggers or Explorers in the List
        :retuns: List of all the Analyzers in the `AnalyzerList`.
        :rtype: `MetaLogger` or `MetaExplorer`
        """
        return iter(self._analyzers)
