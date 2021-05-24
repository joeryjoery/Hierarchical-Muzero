import typing
import sys
from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import gym
import tqdm

from ..interface import Agent
from mazelab_experimenter.utils import find, MazeObjects, rand_argmax


class TabularHierarchicalAgent(Agent, ABC):

    def __init__(self, observation_shape: typing.Tuple, n_actions: int, n_levels: int, **kwargs) -> None:
        super().__init__(observation_shape=observation_shape, n_actions=n_actions)
        # Number of hierarchies inside the agent.
        self.n_levels = n_levels

        # Keep a memory of sampled goals as a stack.
        self._level_goals = np.full(n_levels, None)
        self._goal_steps = np.zeros(n_levels, dtype=np.int32)

    def set_goal(self, goal: int, level: int = -1) -> None:
        self._level_goals[int(level)] = int(goal)

    def clear_hierarchy(self, level: int = 0) -> None:
        if 0 <= level < self.n_levels:
            self._level_goals[:(level + 1)] = None
            self._goal_steps[:(level + 1)] = 0

    @abstractmethod
    def get_level_action(self, s: int, g: int, level: int, **kwargs) -> int:
        """Sample a **Hierarchy action** (not an Environment action) from the Agent. """


class HierQ(TabularHierarchicalAgent):
    """ A basic/ direct reimplementation of the Tabular Hierarchical Q-learning agent from (Levy et al., 2019)

    The functionality adheres as strictly as possible (neglecting some minor mistakes)
    to the pseudocode in Algorithm 2 Appendix 7.2.

    This code was adapted and reformatted from https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-
    """

    def __init__(self, observation_shape: typing.Tuple, n_actions: int, n_levels: int,
                 horizons: typing.Union[int, typing.List[int]], lr: float = 0.5, epsilon: float = 0.1,
                 discount: float = 0.95) -> None:
        super().__init__(observation_shape=observation_shape, n_actions=n_actions, n_levels=n_levels)

        assert type(horizons) == int or (type(horizons) == list and len(horizons) == n_levels), \
            f"Incorrect specification of horizon lengths, " \
            f"this should either be a fixed integer or a list in ascending hierarchy order."

        # Initialize the agent's state space and goal space
        self.S = np.arange(np.prod(observation_shape))
        self.G = [self.S] * n_levels

        # Training and agent hyperparameters.
        self.lr = lr
        self.epsilon = epsilon
        self.discount = discount
        self.horizons = np.full(n_levels, horizons, dtype=np.int32) \
            if type(horizons) == int else np.asarray(horizons, dtype=np.int32)

        # Initialize Q-tables at all levels pessimistically with value '-Horizon_i'.
        self._q_init = -np.clip([np.prod(self.horizons[:(i + 1)]) for i in range(self.n_levels)], 0, max(horizons))
        self._q_tables = [
                             np.full((len(self.S), len(self.G[0]), self.n_actions), self._q_init[0],
                                     dtype=np.float32)
                         ] + [
                             np.full((len(self.S), len(self.G[i + 1]), len(self.G[i])), v, dtype=np.float32)
                             for i, v in enumerate(self._q_init[1:])
                         ]
        # Previous states array keeps track of positions during training for each level i, 0 < i <= n_levels
        self._previous_states = [deque(maxlen=int(np.prod(self.horizons[:(i + 1)]))) for i in range(self.n_levels - 1)]

    def update_critic(self, lvl: int, s: int, a: int, s_next: int, goals: typing.List) -> None:
        # Update Q-Tables with given transition
        ys = (goals != s_next) * (-1 + self.discount * np.amax(self._q_tables[lvl][s_next, goals], axis=-1))
        self._q_tables[lvl][s, goals, a] += self.lr * (ys - self._q_tables[lvl][s, goals, a])

    @staticmethod
    def _get_pos(state: np.ndarray) -> np.ndarray:
        return find(a=state, predicate=lambda x: x == MazeObjects.AGENT.value)

    def _get_index(self, coord: typing.Tuple) -> int:
        return int(np.ravel_multi_index(coord, dims=self.observation_shape))

    def reset(self, full_reset: bool = False) -> None:
        self.clear_hierarchy(self.n_levels - 1 - int(not full_reset))
        for table, value in zip(self._q_tables, self._q_init):
            table[...] = value

    def get_level_action(self, s: int, g: int, level: int, explore: bool = True) -> int:
        """Sample a **Hierarchy action** (not an Environment action) from the Agent. """
        greedy = int(level or not explore or np.random.random_sample() > self.epsilon)
        return rand_argmax(self._q_tables[level][s][g] * greedy, preference=(g if level else None))

    def sample(self, state: np.ndarray, behaviour_policy: bool = True) -> int:
        """Sample an **Environment action** (not a goal) from the Agent. """
        assert self._level_goals[-1] is not None, "No environment goal is specified."

        s = self._get_index(self._get_pos(state))  # Ravelled state coordinate

        # Check if a goal has been achieved an horizon exceeded, and reset accordingly except the top-level.
        achieved = np.flatnonzero(s == self._level_goals)
        exceeded = np.flatnonzero(self._goal_steps[:-1] >= self.horizons[:-1])
        if len(achieved) or len(exceeded):
            self.clear_hierarchy(np.max(achieved.tolist() + exceeded.tolist()))

        # Sample a new action and new goals by iterating and sampling actions from the top to the bottom policy.
        for lvl in reversed(range(self.n_levels)):
            if (lvl == 0) or (self._level_goals[lvl - 1] is None):
                # Get the (behaviour) policy action according to the current level.
                a = self.get_level_action(s=s, g=self._level_goals[lvl], level=lvl, explore=behaviour_policy)
                self._goal_steps[lvl] += 1

                if lvl > 0:  # Set sampled action as goal for the 1-step lower level policy
                    self.set_goal(goal=a, level=lvl - 1)
                else:  # Return atomic action.
                    return a

    def update(self, _env: gym.Env, level: int, state: int, goal_stack: typing.List[int]) -> typing.Tuple[int, bool]:
        """Train level function of Algorithm 2 HierQ by (Levy et al., 2019).
        """
        step, done = 0, False
        while (step < self.horizons[level]) and (state not in goal_stack) and (not done):
            # Sample an action at the current hierarchy level.
            a = self.get_level_action(s=state, g=goal_stack[-1], level=level)

            if level:   # Temporally extend action as a goal.
                s_next, done = self.update(_env=_env, level=level - 1, state=state, goal_stack=goal_stack + [a])
            else:  # Atomic level: Perform a step in the actual environment, and perform critic-table updates.
                next_obs, r, done, meta = _env.step(a)
                s_next = self._get_index(meta['coord_next'])

                # Keep a memory of observed states to update Q-tables with hindsight.
                if state != s_next:
                    for d in self._previous_states:
                        d.append(state)

                # At each level, update Q-tables for each trailing state for all possible subgoals.
                self.update_critic(0, state, a, s_next, self.G[0])
                for i in range(1, self.n_levels):
                    for s_mem in self._previous_states[i - 1]:
                        self.update_critic(i, s_mem, s_next, s_next, self.G[i])

            # Update state of control.
            state = s_next

        return state, done

    def train(self, _env: gym.Env, num_episodes: int, progress_bar: bool = False, **kwargs) -> None:
        """Train the HierQ agent, note that HierQ organizes the training procedure mostly recursively in the `self.update` function.
        """
        iterator = (tqdm.trange(num_episodes, file=sys.stdout, desc="HierQ Training")
                    if progress_bar else range(num_episodes))

        for _ in iterator:
            # Reinitialize environment after each episode.
            state = self._get_index(self._get_pos(_env.reset()))

            # Set top hierarchy goal/ environment goal (only one goal-state is supported for now)
            goal_stack = [self._get_index(_env.unwrapped.maze.get_end_pos()[0])]

            # Reset previous states array for each level i, 0 < i <= n_levels
            for d in self._previous_states:
                d.clear()

            done = False
            while (not done) and (state != goal_stack[0]):
                # Sample a goal as a temporally extended action and observe UMDP transition until env termination.
                a = self.get_level_action(s=state, g=goal_stack[-1], level=self.n_levels - 1)
                state, done = self.update(
                    _env=_env, level=self.n_levels - 2, state=state, goal_stack=goal_stack + [a])

            # Cleanup environment variables
            _env.close()
