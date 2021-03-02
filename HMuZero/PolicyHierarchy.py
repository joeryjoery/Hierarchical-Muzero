from abc import ABC, abstractmethod

import numpy as np

from Interface import SearchBase
from MuZero.MuMCTS import MuZeroContinuousMCTS as CMuMCTS
from MuZero.MuMCTS import MuZeroMCTS
from utils import DotDict
from utils.game_utils import GameState
from utils.HierarchicalUtils import GoalState, HierarchicalGameHistory


class PolicyHierarchy(SearchBase):

    def __init__(self, game, neural_net, args: DotDict) -> None:
        """
        Initialize all requisite variables for performing planning in a Hierarchical setting.
        :param game: Game Implementation of Game class for environment logic.
        :param neural_net: Implementation of the neural networks for inference.
        :param args: DotDict Data structure containing parameters for the tree search.
        """
        super().__init__(game, neural_net, args)

        self.goal_scaler = lambda g: g * (game.obs_high - game.obs_low) + game.obs_low

        # Algorithm specification/ selection. True => MuZero. False => Model-Free
        self.goal_engine = CMuMCTS(game, neural_net.goal_net, args) if self.args.plan_goals else None
        if game.continuous:
            self.action_engine = CMuMCTS(game, neural_net.action_net, args) if self.args.plan_actions else None
        else:
            self.action_engine = MuZeroMCTS(game, neural_net.action_net, args) if self.args.plan_actions else None

    def refresh(self) -> None:
        if self.goal_engine is not None:
            self.goal_engine.refresh()
        if self.action_engine is not None:
            self.action_engine.refresh()

    def goal_achieved(self, current_state: np.ndarray, current_goal: np.ndarray) -> bool:
        """ """
        # TODO: args options for distance based or binary rewards. --> integrate function return as reward value.
        return np.linalg.norm(current_state - current_goal) <= self.args.goal_error

    def sample_goal(self, state: GameState, trajectory: HierarchicalGameHistory, temp: float = 1.0) -> GoalState:
        if self.goal_engine is not None:
            actions, pi, v = self.goal_engine.runMCTS(state, trajectory, temp)
            g = actions[np.random.choice(len(actions), p=pi)]
            goal = GoalState(self.goal_scaler(g), 0, False, len(trajectory))  # TODO
            inference_statistics = (actions, pi, v)
        else:
            # Simply sample a goal
            action = self.neural_net.goal_net.sample(state.observation)
            goal = GoalState(action, 0, False, len(trajectory))
            inference_statistics = ()

        # cast to channel
        goal.goal = goal.goal.reshape(1, 1, -1)

        return goal, inference_statistics

    def sample_action(self, state: GameState, trajectory: HierarchicalGameHistory, goal: GoalState,
                      exploration: float = 1.0):
        self.neural_net.action_net.set_goal(goal)
        if self.action_engine is not None:
            actions, pi, v = self.action_engine.runMCTS(state, trajectory, exploration)
            action = actions[np.random.choice(len(actions), p=pi)]
            inference_statistics = (actions, pi, v)
        else:
            # Sample action using goal conditioned policy
            action = self.neural_net.action_net.sample(state.observation, exploration)
            inference_statistics = ()

        return action, inference_statistics
