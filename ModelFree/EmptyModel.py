"""
Model wrapper to be used in HierarchicalCoach and Network/ Policy Hierarchy as a goal sampling network.

This wrapper only samples null goals for debugging the lower-level policy.
"""
import typing

import numpy as np

from ModelFree.Interface import IUVFA
from utils.HierarchicalUtils import get_goal_space, GoalState


class NullModel(IUVFA):
    """ Implement the Model-Free agent interface for an empty goal sampling agent. """

    def __init__(self, game, net_args) -> None:
        super().__init__(game, net_args, lambda *vargs: None)

        self.monitor = None

        # Set default goal to zeros ==> defaults to no goal/ the canonical algorithm.
        self.set_goal(GoalState(np.zeros(get_goal_space(net_args, game)), 0, False, 0))

    def train(self, examples: typing.List):
        pass

    def sample(self, observation: np.ndarray, **kwargs):
        return self.current_goal

    def save_checkpoint(self, folder: str = 'checkpoint', filename: str = 'checkpoint.pth.tar') -> None:
        pass  # Do nothing

    def load_checkpoint(self, folder: str = 'checkpoint', filename: str = 'checkpoint.pth.tar') -> None:
        pass  # Do nothing



