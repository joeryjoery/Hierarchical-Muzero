import typing
from abc import ABC, abstractmethod

import gym
import numpy as np


class Agent(ABC):
    """ Basic Agent interface, inherit this class to implement algorithms compatible with the experimentation API. """
    _NEUMANN_MOTION: int = 1
    _MOORE_MOTION: int = 2

    def __init__(self, observation_shape: typing.Tuple, n_actions: int) -> None:
        """ Initialize root with domain dimensions. 
        :param observation_shape: tuple Indicates the dimensionality of the environment's observations.
        :param n_actions: int The environment's action space dimension.
        """
        self.observation_shape = observation_shape
        self.n_actions = n_actions
        self.motion = self._NEUMANN_MOTION if n_actions == 4 else self._MOORE_MOTION
    
    @abstractmethod
    def reset(self) -> None:
        """ Refreshes all stateful variables/ reinitializes the agent. """

    @abstractmethod
    def sample(self, state: np.ndarray, **kwargs) -> typing.Union[int, float, np.ndarray]:
        """ Sample an action given an environment context (state) and return an action. """
        
    @abstractmethod
    def update(self, **kwargs) -> None:
        """ Update the agent's internal state/ adapt its policy for learning. """
        
    def train(self, _env: gym.Env, **kwargs) -> None:
        """ Base method definition for training an Agent transparently to the user. 
        
        The functionality of this method depends on the derived class and does not need to be inherited. 
        :param _env: gym.Env The environment to train your agent on.
        """
        pass