from abc import ABC, abstractmethod
import typing

import numpy as np

import Agents
from utils.HierarchicalUtils import GoalState
from utils.debugging import ModelFreeMonitor


class IModelFree(ABC):

    def __init__(self, game, net_args, builder: typing.Callable):
        self.net_args = net_args
        self.network = builder(game, net_args)

        self.monitor = ModelFreeMonitor(self)
        self.steps = 0

    @abstractmethod
    def train(self, examples: typing.List):
        """ """

    @abstractmethod
    def sample(self, observation: np.ndarray, **kwargs):
        """ """

    @abstractmethod
    def save_checkpoint(self, folder: str = 'checkpoint', filename: str = 'checkpoint.pth.tar') -> None:
        """
        Saves the current neural network (with its parameters) in folder/filename
        Each individual part of the MuZero algorithm is stored separately (representation, dynamics, prediction
        and optionally the latent state decoder).

        If specified folder does not yet exists, the method creates a new folder if permitted.

        :param folder: str Path to model weight files
        :param filename: str Base name for model weight files
        """

    @abstractmethod
    def load_checkpoint(self, folder: str = 'checkpoint', filename: str = 'checkpoint.pth.tar') -> None:
        """
        Loads parameters of each neural network model from given folder/filename

        :param folder: str Path to model weight files
        :param filename: str Base name of model weight files
        :raises: FileNotFoundError if one of the three implementations are missing or if path is incorrectly specified.
        """


class IUVFA(IModelFree, ABC):
    current_goal: GoalState = None

    def __init__(self, game, net_args, architecture: typing.Union[str, typing.Callable], goal_sampling: bool = False):
        if isinstance(architecture, str):
            if goal_sampling:
                super().__init__(game, net_args, Agents.ModelFreeNetworks[f'Goal_{architecture}'])
            else:
                super().__init__(game, net_args, Agents.ModelFreeNetworks[f'GC_{architecture}'])
        else:
            super().__init__(game, net_args, architecture)

    def set_goal(self, goal: GoalState) -> None:
        assert isinstance(goal, GoalState)
        self.current_goal = goal

