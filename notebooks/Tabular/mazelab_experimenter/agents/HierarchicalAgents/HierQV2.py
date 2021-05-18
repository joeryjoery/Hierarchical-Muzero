"""
This file implements the canonical Hierarchical Q-learning algorithm by (Levy et al., 2019) along with
a relative action/ sub-goal space encoding, optionality for training goal-conditioned top-level policies,
and the optionality to opt between a shortest-path or binary goal-dependent reward function.

Note that the discount parameter should be in (0, 1) for the binary reward function and (0, 1] for the shortest path.

"""
from __future__ import annotations
import typing
import sys
from collections import deque

import numpy as np
import gym
import tqdm

from ..interface import Agent
from ..HierQ import TabularHierarchicalAgent

from .utils import CriticTable

from mazelab_experimenter.utils import find, MazeObjects, rand_argmax, get_pos
from mazelab_experimenter.utils import ravel_moore_index, ravel_neumann_index, unravel_moore_index, \
    unravel_neumann_index, manhattan_distance, chebyshev_distance, neumann_neighborhood_size, moore_neighborhood_size


class HierQV2(TabularHierarchicalAgent):
    _ILLEGAL: int = -1

    def __init__(self, observation_shape: typing.Tuple, n_actions: int, n_levels: int,
                 horizons: typing.Union[typing.List[int]], lr: float = 0.5, epsilon: float = 0.1,
                 discount: float = 0.95, relative_actions: bool = False, relative_goals: bool = False,
                 universal_top: bool = False, shortest_path_rewards: bool = True,
                 legal_states: np.ndarray = None) -> None:
        super().__init__(observation_shape=observation_shape, n_actions=n_actions, n_levels=n_levels)

        assert type(horizons) == int or (type(horizons) == list and len(horizons) == n_levels), \
            f"Incorrect specification of horizon lengths, " \
            f"this should either be a fixed integer or a list in ascending hierarchy order."

        # Initialize the agent's state space and goal space
        self.S, self.S_legal = np.arange(np.prod(observation_shape)), legal_states
        self.G = [self.S] * n_levels
        self.A = [np.arange(self.n_actions)] + self.G[:-1]  # Defaults (A \equiv G \equiv S for pi_i, i > 0).

        # Cache coordinates of all states and goals for mapping action-deltas.
        self.S_xy = np.asarray(np.unravel_index(self.S, self.observation_shape)).T
        self.G_xy = [self.S_xy] * n_levels

        if relative_actions:  # (code is more convoluted than necessary for debugging purposes)
            # Set goals to the tiles centered around each state with radius_i = horizon_i (TOP level remains absolute)
            self.G_radii = [np.prod(horizons[:(i + 1)]) for i in range(n_levels - 1)]
            self.A[1:] = [np.arange(
                neumann_neighborhood_size(r) if self.motion == Agent._NEUMANN_MOTION else moore_neighborhood_size(r)
            ) for r in self.G_radii]

            # Cache delta xy coordinates (displacements relative to states) for goal sampling action spaces.
            self.A_dxy = [None] + [(unravel_neumann_index(self.A[i + 1], radius=self.G_radii[i], delta=True)
                                    if self.motion == Agent._NEUMANN_MOTION else
                                    unravel_moore_index(self.A[i + 1], radius=self.G_radii[i], delta=True))
                                   for i in range(self.n_levels - 1)]

            # Relative Action map to State space index for each level: f_i(A | S) -> S
            self.A_to_S = [None]
            for i in range(1, self.n_levels):
                Ai_to_S = list()
                for center in self.S_xy:
                    coords = center + self.A_dxy[i]

                    mask = np.all((0, 0) <= coords, axis=-1) & np.all(coords < observation_shape, axis=-1)
                    mask[len(coords) // 2] = 0  # Do nothing action.

                    states = HierQV2._ILLEGAL * np.ones(len(coords))
                    states[mask] = self._get_index(coords[mask].T)

                    Ai_to_S.append(states.astype(np.int32))
                self.A_to_S.append(Ai_to_S)

            # With multiple hierarchies, the action horizon grows super exponentially.
            # If the subgoal-radius starts to exceed the environment dimensions, it is better to use absolute goals.
            if np.sum(np.asarray(self.G_radii) == max(self.observation_shape) // 2 - 1) > 1:
                print("Warning: multiple action spaces exceed the environment' dimensions, perhaps use absolute goals?")

        if relative_goals:
            assert (not universal_top), "Relative goals on the top hierarchy cannot be defined."

            if not relative_actions:
                print("Warning: Opted for relative goals without relative actions, "
                      "even though the latter would also imply the former. This will impact performance.")

            # Set the goal space of level i to the action space of level i+1.
            self.G[:-1] = self.A[1:]
            self.S_to_G = self.A_to_S[1:]

        if not universal_top:
            # Do not train a goal-conditioned top-level policy (more memory efficient).
            self.G[-1] = [0]

        # Training and agent hyperparameters.
        self.lr = lr
        self.epsilon = epsilon
        self.discount = discount
        self.relative_actions = relative_actions
        self.relative_goals = relative_goals
        self.universal_top = universal_top
        self.shortest_path_rewards = shortest_path_rewards
        self.horizons = np.full(n_levels, horizons, dtype=np.int32) \
            if type(horizons) == int else np.asarray(horizons, dtype=np.int32)

        # Initialize Q-tables at all levels pessimistically with value '-Horizon_i'.
        self.critics = list()
        for i in range(self.n_levels):
            init = -np.clip(np.prod(self.horizons[:(i + 1)]), 0, max(horizons))
            init *= int(self.shortest_path_rewards)  # Init with 0's if not shortest path rewards.
            dims = (len(self.S), len(self.G[i]), len(self.A[i]))

            critic = CriticTable(init, dims, goal_conditioned=(i < self.n_levels - 1 or self.universal_top))
            critic.reset()

            self.critics.append(critic)

        # Previous states array keeps track of positions during training for each level i, 0 < i <= n_levels
        self.atomic_horizons = [int(np.prod(self.horizons[:i])) for i in range(0, self.n_levels + 1)]
        self._previous_states = [deque(maxlen=self.atomic_horizons[i]) for i in range(1, self.n_levels)]

    @staticmethod
    def ravel_delta_indices(center_deltas: np.ndarray, r: int, motion: int) -> np.ndarray:
        if motion == Agent._MOORE_MOTION:
            return ravel_moore_index(center_deltas, radius=r, delta=True)
        else:
            return ravel_neumann_index(center_deltas, radius=r, delta=True)

    def inside_radius(self, a: np.ndarray, b: np.ndarray, r: int) -> bool:
        """ Check whether the given arrays 'a' and 'b' are contained within the radius dependent on the motion. """
        return (manhattan_distance(a, b) if self.motion == Agent._NEUMANN_MOTION else chebyshev_distance(a, b)) < r

    def goal_reachable(self, level: int, state: int, goal: int) -> bool:
        if not self.relative_goals:
            return True
        return self.inside_radius(self.S_xy[state], self.S_xy[goal], self.atomic_horizons[level])

    def _get_index(self, coord: typing.Tuple, dims: typing.Optional[typing.Tuple[int, int]] = None) -> int:
        return np.ravel_multi_index(coord, dims=self.observation_shape if dims is None else dims)

    def reset(self, full_reset: bool = False) -> None:
        self.clear_hierarchy(self.n_levels - 1 - int(not full_reset))
        for critic in self.critics:
            critic.reset()

    def update_critic(self, level: int, s: int, a: int, s_next: int, end_pos: int = None) -> None:
        """Update Q-Tables with given transition """

        if level and self.relative_actions:  # Convert subgoal-action 'a' from Absolute-to-Neighborhood.
            delta = np.asarray((self.S_xy[s], self.S_xy[a])).T
            a = self.ravel_delta_indices(np.diff(delta).T, r=self.G_radii[level - 1], motion=self.motion).item()

        # Hindsight action transition mask.
        if self.relative_goals:
            delta = np.asarray((self.S_xy[s], self.S_xy[s_next])).T
            g_her = self.ravel_delta_indices(np.diff(delta).T, r=self.atomic_horizons[level], motion=self.motion).item()

            goals = self.G[level]
            mask = np.zeros_like(goals)
            if self.critics[level].goal_conditioned:
                mask[g_her] = 1
            else:  # Non-universal top level
                mask[0] = (end_pos == s_next)
        else:
            goals, mask = (self.G[level], self.G[level] == s_next) \
                if self.critics[level].goal_conditioned else (0, end_pos == s_next)

        # Q-Learning update for each goal-state.
        if self.shortest_path_rewards:
            ys = (1 - mask) * (-1 + self.discount * np.amax(self.critics[level].table[s_next, goals], axis=-1))
        else:
            ys = mask + (1 - mask) * self.discount * np.amax(self.critics[level].table[s_next, goals], axis=-1)
        self.critics[level].table[s, goals, a] += self.lr * (ys - self.critics[level].table[s, goals, a])

    def get_level_action(self, s: int, g: int, level: int, explore: bool = True) -> int:
        """Sample a **Hierarchy action** (not an Environment action) from the Agent. """
        # Helper variables
        greedy = int(level or (not explore) or (np.random.random_sample() > self.epsilon))
        gc = g * int(self.critics[level].goal_conditioned)  # TODO: g to relative index

        pref = g if level else None        # Goal-selection bias for shared max Q-values.
        if level and self.relative_actions:  # Action Preference: Absolute-to-Neighborhood for policy Action-Space
            pref = None
            if self.inside_radius(self.S_xy[s], self.S_xy[g], r=self.G_radii[level - 1]):
                pref = self.ravel_delta_indices(center_deltas=np.diff([self.S_xy[s], self.S_xy[g]]).T,
                                                r=self.G_radii[level - 1], motion=self.motion).item()

        mask = np.ones_like(self.critics[level].table[s, gc], dtype=bool)
        if level:  # Prune/ correct action space by masking out-of-bound or illegal subgoals.
            if self.relative_actions:
                mask = (self.A_to_S[level][s] != HierQV2._ILLEGAL)
            else:
                mask[s] = 0  # Do nothing is illegal.
            if self.S_legal is not None:
                pass  # TODO Illegal move masking.

        # Sample an action (with preference for the end-goal if goal-conditioned policy).
        action = rand_argmax(self.critics[level].table[s, gc] * greedy, preference=pref, mask=mask)

        if level and self.relative_actions:  # Neighborhood-to-Absolute for sampled action.
            action = self.A_to_S[level][s][action]

        return action

    def sample(self, state: np.ndarray, behaviour_policy: bool = True) -> int:
        """Sample an **Environment action** (not a goal) from the Agent. """
        assert self._level_goals[-1] is not None, "No environment goal is specified."

        s = self._get_index(get_pos(state))  # Ravelled state coordinate

        # Check if a goal has been achieved an horizon exceeded, and reset accordingly except the top-level.
        achieved = np.flatnonzero(s == self._level_goals)
        exceeded = np.flatnonzero(self._goal_steps[:-1] >= self.horizons[:-1])
        reachable = np.flatnonzero([(not self.goal_reachable(i, s, self._level_goals[i]))
                                    for i in range(self.n_levels-1)][::-1])

        if len(achieved) or len(exceeded) or len(reachable):
            self.clear_hierarchy(np.max(achieved.tolist() + exceeded.tolist() + reachable.tolist()))

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
                if state != s_next:
                    for d in self._previous_states:
                        d.append(state)

                # At each level, update Q-tables for each trailing state for all possible subgoals.
                # print(state, a, s_next)
                self.update_critic(0, state, a, s_next)
                for i in range(1, self.n_levels):
                    for s_mem in self._previous_states[i - 1]:
                        self.update_critic(i, s_mem, s_next, s_next, end_pos=goal_stack[0])

            # Update state of control. Terminate level if new state is out of reach of current goal.
            state = s_next
            step += 1
            if not self.goal_reachable(level, state, goal_stack[-1]):
                break

        return state, done

    def train(self, _env: gym.Env, num_episodes: int, progress_bar: bool = False, **kwargs) -> None:
        """Train the HierQ agent through recursion with the `self.update` function.
        """
        iterator = (tqdm.trange(num_episodes, file=sys.stdout, desc="HierQ Training")
                    if progress_bar else range(num_episodes))

        for _ in iterator:
            # Reinitialize environment after each episode.
            state = self._get_index(get_pos(_env.reset()))

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
