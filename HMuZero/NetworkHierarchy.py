"""
Defines the base structure of a neural network for MuZero. Provides a generic implementation
for performing loss computation and recurrent unrolling of the network given data.

Notes:
 - Base implementation done.
 - Documentation 14/11/2020
"""
import typing

from utils import DotDict
from utils.debugging import HierarchicalMuZeroMonitor


class TwoLevelNetworkHierarchy:
    """
    Wrapper class for holding two different policy instances.
    """

    def __init__(self, game, net_args: DotDict, builder: typing.Callable) -> None:
        """
        Initialize base MuZero Neural Network. Contains all requisite logic to work with any
        MuZero network and environment.
        :param game: Implementation of base Game class for environment logic.
        :param net_args: DotDict Data structure that contains all neural network arguments as object attributes.
        :param builder: Wrapper function that takes the game and net_args as parameters and returns two classes.
        :raises: NotImplementedError if invalid optimization method is specified in the provided .json configuration.
        """
        self.architecture = builder
        self.fit_rewards = (game.n_players == 1)
        self.net_args = net_args

        self.monitor = HierarchicalMuZeroMonitor(self)

        # Builder returns 2x (typing.Union[IUVFA, GoalConditionedMuZero])
        self.goal_net, self.action_net = builder(game, net_args)
        self.steps = 0

    def train(self, examples: typing.List) -> None:
        """
        This function trains the neural network with data gathered from self-play.

        :param examples: a list of training examples of the form. TODO: doc
        """
        goal_examples, act_examples = examples

        self.goal_net.train(goal_examples)
        self.action_net.train(act_examples)
        self.steps += 1

    def save_checkpoint(self, folder: str = 'checkpoint', filename: str = 'checkpoint.pth.tar') -> None:
        """
        Saves the current neural network (with its parameters) in folder/filename
        Each individual part of the MuZero algorithm is stored separately (representation, dynamics, prediction
        and optionally the latent state decoder).

        If specified folder does not yet exists, the method creates a new folder if permitted.

        :param folder: str Path to model weight files
        :param filename: str Base name for model weight files
        """
        self.action_net.save_checkpoint(folder, f'action_{filename}')
        self.goal_net.save_checkpoint(folder, f'goal_{filename}')

    def load_checkpoint(self, folder: str = 'checkpoint', filename: str = 'checkpoint.pth.tar') -> None:
        """
        Loads parameters of each neural network model from given folder/filename

        :param folder: str Path to model weight files
        :param filename: str Base name of model weight files
        :raises: FileNotFoundError if one of the three implementations are missing or if path is incorrectly specified.
        """
        self.action_net.load_checkpoint(folder, f'action_{filename}')
        self.goal_net.load_checkpoint(folder, f'goal_{filename}')
