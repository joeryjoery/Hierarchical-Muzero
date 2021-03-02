from abc import ABC, abstractmethod

from utils import DotDict


class SearchBase(ABC):

    def __init__(self, game, neural_net, args: DotDict) -> None:
        """
        Initialize all requisite variables for performing the implemented search algorithm.
        :param game: Game Implementation of Game class for environment logic.
        :param neural_net: NeuralNetwork class for inference.
        :param args: DotDict Data structure containing parameters for the tree search.
        """
        self.game = game
        self.neural_net = neural_net
        self.args = args

    @abstractmethod
    def refresh(self) -> None:
        """ Clear all statistics/ data kept in the current search memory/ history. """
