"""
Initialization module to define constructor classes/ agent implementations in the 'Agents' scope.

To add a new neural network, add a key-argument to the AlphaZeroNetworks or MuZeroNetworks dictionary with as
value the class reference that constructs the neural network.
"""
from functools import partial

from .GymNetwork import AlphaZeroGymNetwork, MuZeroGymNetwork, ContinuousMuZeroGymNetwork, \
    GoalConditionedMuZeroGymNetwork, GoalConditionedContinuousMuZeroGymNetwork, DQNGymNetwork, DDPGGymNetwork
from .AtariNetwork import AlphaZeroAtariNetwork, MuZeroAtariNetwork
from .HexNetwork import AlphaZeroHexNetwork, MuZeroHexNetwork
from .Player import Player, ManualPlayer, RandomPlayer, DeterministicPlayer, \
    DefaultMuZeroPlayer, DefaultAlphaZeroPlayer, BlindMuZeroPlayer, HierarchicalMuZeroPlayer, ModelFreePlayer


# Add your AlphaZero neural network architecture here by referencing the imported Class with a string key.
AlphaZeroNetworks = {
    'Hex': AlphaZeroHexNetwork,
    'Othello': AlphaZeroHexNetwork,
    'Gym': AlphaZeroGymNetwork,
    "Atari": AlphaZeroAtariNetwork
}


# Add your MuZero neural network architecture here by referencing the imported Class with a string key.
MuZeroNetworks = {
    'Hex': MuZeroHexNetwork,
    'Othello': MuZeroHexNetwork,
    'GC_Gym': GoalConditionedMuZeroGymNetwork,
    'Gym': MuZeroGymNetwork,
    'Atari': MuZeroAtariNetwork
}


# Add your MuZero neural network architecture for Continuous Action spaces. This reuses default architectures
# but redefines the output parameterization. This is split from 'MuZeroNetworks' for compatibility clashes.
ContinuousMuZeroNetworks = {
    'Gym': ContinuousMuZeroGymNetwork,
    'GC_Gym': GoalConditionedContinuousMuZeroGymNetwork,
    'Goal_Gym': partial(GoalConditionedContinuousMuZeroGymNetwork, goal_sampling=True)
}

ModelFreeNetworks = {
    'Goal_DDPG_Gym': partial(DDPGGymNetwork, goal_sampling=True),
    'GC_DDPG_Gym': DDPGGymNetwork,
    'GC_DQN_Gym': DQNGymNetwork
}


# Add different agent implementations for interacting with environments.
Players = {
    "ALPHAZERO": DefaultAlphaZeroPlayer,
    "MUZERO": DefaultMuZeroPlayer,
    "BLIND_MUZERO": BlindMuZeroPlayer,
    "RANDOM": RandomPlayer,
    "DETERMINISTIC": DeterministicPlayer,
    "MANUAL": ManualPlayer
}
