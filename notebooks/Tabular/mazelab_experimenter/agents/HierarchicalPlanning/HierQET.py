from __future__ import annotations
import typing
import sys
from dataclasses import dataclass
from collections import deque

import numpy as np
import gym
import tqdm

from ..interface import Agent
from ..HierQ import TabularHierarchicalAgent

from mazelab_experimenter.utils import find, MazeObjects, rand_argmax
from mazelab_experimenter.utils import ravel_moore_index, ravel_neumann_index, unravel_moore_index, \
    unravel_neumann_index, manhattan_distance, chebyshev_distance


class HierQET(TabularHierarchicalAgent):

    @dataclass
    class GoalTransition:
        state: np.ndarray
        goal: typing.Union[int, np.ndarray]  # Singular goal or multiple (HER)
        action: int
        next_state: np.ndarray
        reward: typing.Optional[float]  # Rewards can be extrinsic or derived from (state == goal).

    @dataclass
    class CriticTable:
        init: float
        dimensions: typing.Tuple
        goal_conditioned: bool = True
        table: np.ndarray = None

        def reset(self):
            if self.table is None:
                self.table = np.zeros(self.dimensions)
            self.table[...] = self.init

    def __init__(self, observation_shape: typing.Tuple, n_actions: int, n_levels: int,
                 horizons: typing.Union[typing.List[int]], lr: float = 0.5, epsilon: float = 0.1,
                 discount: float = 0.95, n_tb: int = 1, universal_top: bool = False,
                 ignore_training_time: bool = False, legal_states: np.ndarray = None) -> None:
        super().__init__(observation_shape=observation_shape, n_actions=n_actions, n_levels=n_levels)

        assert type(horizons) == int or (type(horizons) == list and len(horizons) == n_levels), \
            f"Incorrect specification of horizon lengths, " \
            f"this should either be a fixed integer or a list in ascending hierarchy order."

        # Initialize the agent's state space and goal space
        self.S, self.S_legal = np.arange(np.prod(observation_shape)), legal_states
        self.G = [self.S] * n_levels
        self.A = [np.arange(self.n_actions)] + self.G[:-1]  # Defaults (A \equiv G \equiv S for pi_i, i > 0).

        # Set goals to the tiles centered around each state with radius_i = horizon_i (TOP level remains absolute)
        self.G_radii = [np.prod(horizons[:(i + 1)]) for i in range(n_levels - 1)]
        self.A[1:] = [np.arange(self._goal_tiles(r, self.motion)) for r in self.G_radii]

        # Cache delta xy coordinates (displacements relative to states) for goal sampling action spaces.
        self.A_dxy = [None] + [self.unravel_delta_indices(self.A[i + 1], self.G_radii[i], self.motion)
                               for i in range(self.n_levels - 1)]

        # Cache coordinates of all states and goals for mapping action-deltas.
        self.S_xy = np.asarray(np.unravel_index(self.S, self.observation_shape)).T
        self.G_xy = [self.S_xy] * n_levels

        # With multiple hierarchies, the action horizon grows super exponentially.
        # If the subgoal-radius starts to exceed the environment dimensions, it is better to use absolute goals.
        if np.sum(np.asarray(self.G_radii) == max(self.observation_shape) // 2 - 1) > 1:
            print("Warning: multiple action spaces exceed the environment' dimensions, perhaps use absolute goals?")

        if not universal_top:
            # Do not train a goal-conditioned top-level policy (bit more memory efficient).
            self.G[-1] = [None]

        # Training and agent hyperparameters.
        self.lr = lr
        self.n_tb = n_tb
        self.epsilon = epsilon
        self.discount = discount
        self.universal_top = universal_top
        self.ignore_training_time = ignore_training_time
        self.horizons = np.full(n_levels, horizons, dtype=np.int32) \
            if type(horizons) == int else np.asarray(horizons, dtype=np.int32)

        # Initialize Q-tables at all levels pessimistically with value '-Horizon_i'.
        self.critics = list()
        for i in range(self.n_levels):
            init = 0.0  #-np.clip(np.prod(self.horizons[:(i + 1)]), 0, max(horizons))
            dims = (len(self.S), len(self.G[i]), len(self.A[i]))

            critic = HierQET.CriticTable(init, dims, goal_conditioned=(i < self.n_levels - 1 or self.universal_top))
            critic.reset()

            self.critics.append(critic)

        # Transition array for updating Q-tables during episodes.
        self._transitions = list()

    @staticmethod
    def _get_pos(observation: np.ndarray) -> np.ndarray:
        """Infer the state-space coordinate of the agent from an environment observation. """
        return find(a=observation, predicate=lambda x: x == MazeObjects.AGENT.value)

    @staticmethod
    def _goal_tiles(radius: int, motion: int) -> int:
        """Get the number of tiles in the Von Neumann neighbourhood or in the Moore neighborhood. """
        return int(radius ** 2 + (radius + 1) ** 2 if motion == Agent._NEUMANN_MOTION else (2 * radius + 1) ** 2)

    @staticmethod
    def ravel_delta_indices(center_deltas: np.ndarray, r: int, motion: int) -> np.ndarray:
        if motion == Agent._MOORE_MOTION:
            return ravel_moore_index(center_deltas, radius=r, delta=True)
        else:
            return ravel_neumann_index(center_deltas, radius=r, delta=True)

    @staticmethod
    def unravel_delta_indices(indices: np.ndarray, radius: int, motion: int) -> np.ndarray:
        if motion == Agent._MOORE_MOTION:
            return unravel_moore_index(indices, radius, delta=True)
        else:
            return unravel_neumann_index(indices, radius, delta=True)

    @staticmethod
    def inside_radius(a: np.ndarray, b: np.ndarray, r: int, motion: int) -> bool:
        """ Check whether the given arrays 'a' and 'b' are contained within the radius dependent on the motion. """
        return (manhattan_distance(a, b) if motion == Agent._NEUMANN_MOTION else chebyshev_distance(a, b)) < r

    def _get_index(self, coord: typing.Tuple, dims: typing.Optional[typing.Tuple[int, int]] = None) -> int:
        dims = self.observation_shape if dims is None else dims
        return int(coord[0] * dims[0] + coord[1])

    def reset(self, full_reset: bool = False) -> None:
        self.clear_hierarchy(self.n_levels - 1 - int(not full_reset))
        for critic in self.critics:
            critic.reset()

    def get_level_action(self, s: int, g: int, level: int, explore: bool = True) -> int:
        """Sample a **Hierarchy action** (not an Environment action) from the Agent. """
        # Helper variables
        greedy = int(not explore or np.random.random_sample() > self.epsilon)
        gc = g * int(self.critics[level].goal_conditioned)

        pref = g if level else None        # Goal-selection bias for shared max Q-values.
        if level:  # Action Preference: Absolute-to-Neighborhood for policy Action-Space
            pref = None
            if HierQET.inside_radius(self.S_xy[s], self.G_xy[level][g], r=self.G_radii[level - 1], motion=self.motion):
                pref = self.ravel_delta_indices(center_deltas=np.diff([self.S_xy[s], self.G_xy[level][g]]).T,
                                                r=self.G_radii[level - 1], motion=self.motion).item()

        mask = np.ones_like(self.critics[level].table[s, gc], dtype=bool)
        if level:  # Prune/ correct action space by masking out-of-bound or illegal subgoals.
            mask[len(mask) // 2] = 0  # Do nothing is illegal.

            c = self.S_xy[s] + self.A_dxy[level]  # Check all coordinates for out of bound cases.
            mask &= np.all((0, 0) <= c, axis=-1) & np.all(c < self.observation_shape, axis=-1)
            if self.S_legal is not None:
                pass  # TODO Illegal move masking.

        # Sample an action (with preference for the end-goal if goal-conditioned policy).
        action = rand_argmax(self.critics[level].table[s, gc] * greedy, preference=pref, mask=mask)

        if level:  # Neighborhood-to-Absolute for sampled action.
            action = self._get_index(self.S_xy[s] + self.A_dxy[level][action])

        return action

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
            if lvl == 0 or self._level_goals[lvl - 1] is None:
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

            if level:  # Temporally extend action as a goal.
                s_next, done = self.update(_env=_env, level=level - 1, state=state, goal_stack=goal_stack + [a])
            else:  # Atomic level: Perform a step in the actual environment, and perform critic-table updates.
                next_obs, r, done, meta = _env.step(a)
                s_next = self._get_index(meta['coord_next'])

                # Keep a memory of observed states to update Q-tables with hindsight.
                self._transitions.append(HierQET.GoalTransition(
                    state=state, goal=goal_stack[0], action=a, next_state=s_next, reward=None)
                )

                # At each level, update Q-tables for each trailing state for all possible subgoals.
                self.update_critic(0, [self._transitions[-1], self._transitions[-1]])  # TODO Tree backup for atomic level
                for i in range(1, self.n_levels):
                    h = np.prod(self.horizons[:i])

                    terminal = s_next == goal_stack[0]
                    for depth in range((self.n_tb - 1) * int(not terminal), self.n_tb):
                        self._trailing_tb(level=i, width=h, depth=depth, path=(len(self._transitions) - 1, ))

            # Update state of control.
            state = s_next
            step += 1

        return state, done

    def _trailing_tb(self, level: int, width: int, depth: int, path: typing.Tuple) -> None:
        # Greedy n-step Tree Backup over trailing states.
        end = path[-1] + 1
        for i in range(min([end, width])):
            idx = end - 1 - i

            if depth <= 0:  # Do update  TODO: Currently == n-step SARSA --> Convert to Tree-Backup
                self.update_critic(level=level, transitions=[self._transitions[x] for x in path + (idx, )][::-1])
            elif idx > 0:  # Recurse down
                self._trailing_tb(level=level, width=width, depth=depth - 1, path=path + (idx - 1, ))
            # Do nothing to prevent a n < self.n Tree Backup.

    def update_critic(self, level: int, transitions: typing.List[HierQET.GoalTransition]) -> None:
        """Update Q-Tables with given transition """
        update_transition = transitions[0]
        end_transition = transitions[-1]

        if not level:
            s, a, s_next = update_transition.state, update_transition.action, update_transition.next_state

            # Hindsight replay. Check which goals have been achieved in the current state.
            goals = self.G[level]
            mask = goals == s_next

            # 1-step Q-return for atomic critic table.   TODO --> n-step return.
            rewards = mask  # r_g(s, a, s') = 1 if s' == g else 0
            ys = rewards + (1 - mask) * self.discount * np.amax(self.critics[level].table[s_next, goals], axis=-1)

            self.critics[level].table[s, goals, a] += self.lr * (ys - self.critics[level].table[s, goals, a])
        else:
            s = update_transition.state
            s_next = transitions[1].next_state
            end_state = end_transition.next_state
            end_pos = end_transition.goal

            delta = np.asarray(np.unravel_index([s, s_next], self.observation_shape))
            a = self.ravel_delta_indices(np.diff(delta).T, r=self.G_radii[level - 1], motion=self.motion).item()

            goals = self.G[level]
            mask = end_state == goals        # Hindsight action transition mask.
            if not self.critics[level].goal_conditioned:
                goals = 0                    # Correct for non-universal critic.
                mask = end_state == end_pos  # Correct hindsight action transition mask.

            r = mask  # r_g(s, a, s') = 1 if s' == g else 0
            ys = r + (1 - mask) * self.discount * np.amax(self.critics[level].table[end_state, goals], axis=-1)
            for _ in range(len(transitions) - 2):
                ys = r + self.discount * ys

            # Q-Learning update for each goal-state.
            self.critics[level].table[s, goals, a] += self.lr * (ys - self.critics[level].table[s, goals, a])

    def train(self, _env: gym.Env, num_episodes: int, progress_bar: bool = False, **kwargs) -> None:
        """Train the HierQ agent through recursion with the `self.update` function.
        """
        iterator = (tqdm.trange(num_episodes, file=sys.stdout, desc="HierQ(n) Training")
                    if progress_bar else range(num_episodes))

        for _ in iterator:
            # Reinitialize environment after each episode.
            state = self._get_index(self._get_pos(_env.reset()))

            # Set top hierarchy goal/ environment goal (only one goal-state is supported for now)
            goal_stack = [self._get_index(_env.unwrapped.maze.get_end_pos()[0])]

            # Reset previous states array for each level i, 0 < i <= n_levels
            self._transitions.clear()

            done = False
            while (not done) and (state != goal_stack[0]):
                # Sample a goal as a temporally extended action and observe UMDP transition until env termination.
                a = self.get_level_action(s=state, g=goal_stack[-1], level=self.n_levels - 1)
                state, done = self.update(
                    _env=_env, level=self.n_levels - 2, state=state, goal_stack=goal_stack + [a])

            # Cleanup environment variables
            _env.close()
