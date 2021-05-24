"""
This file implements the canonical Hierarchical Q-learning algorithm by (Levy et al., 2019) along with
a relative action/ sub-goal space encoding, optionality for training goal-conditioned top-level policies,
and the optionality to opt between a shortest-path or binary goal-dependent reward function.

Note that the discount parameter should be in (0, 1) for the binary reward function and (0, 1] for the shortest path.

"""
from __future__ import annotations
import typing
import sys
from abc import ABC

import numpy as np
import gym
import tqdm

from ..interface import Agent
from .HierQ import TabularHierarchicalAgent

from .utils import CriticTable, GoalTransition, HierarchicalTrace

from mazelab_experimenter.utils import rand_argmax, get_pos
from mazelab_experimenter.utils import ravel_moore_index, ravel_neumann_index, unravel_moore_index, \
    unravel_neumann_index, manhattan_distance, chebyshev_distance, neumann_neighborhood_size, moore_neighborhood_size


class TabularHierarchicalAgentV2(TabularHierarchicalAgent, ABC):
    _ILLEGAL: int = -1

    def __init__(self, observation_shape: typing.Tuple, n_actions: int, n_levels: int,
                 horizons: typing.Union[typing.List[int]], lr: float = 0.5, epsilon: float = 0.1,
                 discount: float = 0.95, relative_actions: bool = False, relative_goals: bool = False,
                 universal_top: bool = False, shortest_path_rewards: bool = False,
                 legal_states: np.ndarray = None, **kwargs) -> None:
        super().__init__(observation_shape=observation_shape, n_actions=n_actions, n_levels=n_levels, **kwargs)

        assert type(horizons) == int or (type(horizons) == list and len(horizons) == n_levels), \
            f"Incorrect specification of horizon lengths, " \
            f"this should either be a fixed integer or a list in ascending hierarchy order."

        # Learning parameters
        self.lr = lr
        self.epsilon = epsilon
        self.discount = discount
        self.horizons = np.full(n_levels, horizons, dtype=np.int32) \
            if type(horizons) == int else np.asarray(horizons, dtype=np.int32)
        self.atomic_horizons = [int(np.prod(self.horizons[:i])) for i in range(0, self.n_levels + 1)]

        # Agent parameterization configuration.
        self.relative_actions = relative_actions
        self.relative_goals = relative_goals
        self.universal_top = universal_top
        self.shortest_path_rewards = shortest_path_rewards

        # Initialize the agent's state space, goal space and action space.
        self.S, self.S_legal = np.arange(np.prod(observation_shape)), legal_states
        self.G = [self.S] * n_levels
        self.A = [np.arange(self.n_actions)] + self.G[:-1]  # Defaults (A \equiv G \equiv S for pi_i, i > 0).

        # Cache coordinates of all states, goals, and actions for fast element-to-coordinate mapping.
        self.S_xy = np.asarray(np.unravel_index(self.S, observation_shape)).T
        self.G_xy = [self.S_xy] * n_levels
        self.A_xy = [None] + [self.S_xy] * (n_levels - 1)

        if self.relative_actions:  # Bound the action space for each policy level proportional to its horizon.
            self.A[1:], self.A_xy[1:] = self._to_neighborhood(radii=self.atomic_horizons[1:-1])

            # Yield a warning if the action-radii start to vastly exceed the environment dimensions.
            if np.sum(np.asarray(self.atomic_horizons[1:-1]) == max(self.observation_shape) // 2 - 1) > 1:
                print("Warning: multiple action spaces exceed the environment' size, perhaps use absolute goals?")

        if relative_goals:  # Bound the goal space for each policy level proportional to its horizon + a small constant.
            c_r = self.horizons[0]  # Goal-radius correction term.
            self.G[:-1], self.G_xy[:-1] = self._to_neighborhood(radii=[(r + c_r) for r in self.atomic_horizons[1:-1]])
            raise NotImplementedError("TODO")

        if not self.universal_top:  # Do not condition top-level policy/ table on goals (more memory efficient).
            self.G[-1] = [0]

        # Given all adjusted dimensions, initialize Q tables.
        self.critics = list()
        for i in range(self.n_levels):
            # Use pessimistic -Horizon_i initialization if using dense '-1' penalties, otherwise use 0s.
            init = -np.clip(self.atomic_horizons[i+1], 0, max(self.horizons)) * int(self.shortest_path_rewards)

            dims = (len(self.S), len(self.G[i]), len(self.A[i]))
            critic = CriticTable(init, dims, goal_conditioned=(i < self.n_levels - 1 or self.universal_top))
            critic.reset()

            self.critics.append(critic)

    def _to_neighborhood(self, radii: typing.List) -> typing.Tuple[typing.List, typing.List]:
        """ Correct the current absolute action space parameterization to a relative/ bounded action space. """
        relative_indices = [np.arange(
            neumann_neighborhood_size(r) if self.motion == Agent._NEUMANN_MOTION else moore_neighborhood_size(r)
        ) for r in radii]

        # Correct action-to-coordinate map to state space index for each level: f_i(A | S) -> S
        relative_coordinates = list()
        for i in range(1, self.n_levels):
            shifts = (   # First gather all relative coordinate displacements for each state.
                unravel_neumann_index(relative_indices[i - 1], radius=radii[i - 1], delta=True)
                if self.motion == Agent._NEUMANN_MOTION else
                unravel_moore_index(relative_indices[i - 1], radius=radii[i - 1], delta=True))

            neighborhood_states = list()
            for center in self.S_xy:
                coords = center + shifts  # All reachable coordinates from state 'center'

                # Mask out out of bound actions.
                mask = np.all((0, 0) <= coords, axis=-1) & np.all(coords < self.observation_shape, axis=-1)
                mask[len(coords) // 2] = 0  # Do nothing action.

                states = TabularHierarchicalAgentV2._ILLEGAL * np.ones(len(coords))
                states[mask] = self._get_index(coords[mask].T)

                neighborhood_states.append(states.astype(np.int32))

            # Override current (absolute) mapping to their corrected displacements.
            relative_coordinates.append(neighborhood_states)

        return relative_indices, relative_coordinates

    def _get_index(self, coord: typing.Tuple, dims: typing.Optional[typing.Tuple[int, int]] = None) -> int:
        return np.ravel_multi_index(coord, dims=self.observation_shape if dims is None else dims)

    def reset(self, full_reset: bool = False) -> None:
        self.clear_hierarchy(self.n_levels - 1 - int(not full_reset))
        for critic in self.critics:
            critic.reset()

    def sample(self, state: np.ndarray, behaviour_policy: bool = True) -> int:
        """Sample an **Environment action** (not a goal) from the Agent. """
        assert self._level_goals[-1] is not None, "No environment goal is specified."

        s = self._get_index(get_pos(state))  # Ravelled state coordinate

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


class HierQV2(TabularHierarchicalAgentV2):

    def __init__(self, observation_shape: typing.Tuple, n_actions: int, n_levels: int,
                 horizons: typing.Union[typing.List[int]], lr: float = 0.5, epsilon: float = 0.1,
                 discount: float = 0.95, relative_actions: bool = False, relative_goals: bool = False,
                 universal_top: bool = False, shortest_path_rewards: bool = False,
                 legal_states: np.ndarray = None, **kwargs) -> None:
        super().__init__(
            observation_shape=observation_shape, n_actions=n_actions, n_levels=n_levels,
            horizons=horizons, lr=lr, epsilon=epsilon, discount=discount,
            relative_actions=relative_actions, relative_goals=relative_goals, universal_top=universal_top,
            shortest_path_rewards=shortest_path_rewards, legal_states=legal_states, **kwargs
        )
        # Data structures for keeping track of an environment trace along with a deque that retains previous
        # states within the hierarchical policies' action horizon. Last atomic_horizon value should be large.
        self.trace = HierarchicalTrace(num_levels=self.n_levels, horizons=self.atomic_horizons)
        self.trace.reset()

    @staticmethod
    def ravel_delta_indices(center_deltas: np.ndarray, r: int, motion: int) -> np.ndarray:
        if motion == Agent._MOORE_MOTION:
            return ravel_moore_index(center_deltas, radius=r, delta=True)
        else:
            return ravel_neumann_index(center_deltas, radius=r, delta=True)

    @staticmethod
    def inside_radius(a: np.ndarray, b: np.ndarray, r: int, motion: int) -> bool:
        """ Check whether the given arrays 'a' and 'b' are contained within the radius dependent on the motion. """
        return (manhattan_distance(a, b) if motion == Agent._NEUMANN_MOTION else chebyshev_distance(a, b)) < r

    def convert_action(self, level: int, reference: int, displaced: int, to_absolute: bool = False) -> int:
        if to_absolute:
            return self.A_xy[level][reference][displaced]
        else:
            return self.ravel_delta_indices((self.S_xy[displaced] - self.S_xy[reference])[None, :],
                                            r=self.atomic_horizons[level], motion=self.motion).item()

    def goal_reachable(self, level: int, state: int, goal: int) -> bool:
        if not self.relative_goals:
            return True
        return self.inside_radius(self.S_xy[state], self.S_xy[goal], r=self.atomic_horizons[level], motion=self.motion)

    def get_level_action(self, s: int, g: int, level: int, explore: bool = True) -> int:
        """Sample a **Hierarchy action** (not an Environment action) from the Agent. """
        # Helper variables
        gc = g * int(self.critics[level].goal_conditioned)  # TODO: g to relative index

        if (not level) and explore:  # Epsilon greedy
            if self.epsilon > np.random.rand():
                return np.random.randint(self.n_actions)

            optimal = np.flatnonzero(self.critics[level].table[s, gc] == self.critics[level].table[s, gc].max())
            return np.random.choice(optimal)

        greedy = int(level or (not explore) or (np.random.rand() > self.epsilon))

        pref = g if level else None          # Goal-selection bias for shared max Q-values.
        if level and self.relative_actions:  # Action Preference: Absolute-to-Neighborhood for policy Action-Space
            pref = None
            if self.inside_radius(self.S_xy[s], self.S_xy[g], r=self.atomic_horizons[level], motion=self.motion):
                pref = self.ravel_delta_indices(center_deltas=np.diff([self.S_xy[s], self.S_xy[g]]).T,
                                                r=self.atomic_horizons[level], motion=self.motion).item()

        mask = np.ones_like(self.critics[level].table[s, gc], dtype=bool)
        if level:  # Prune/ correct action space by masking out-of-bound or illegal subgoals.
            if self.relative_actions:
                mask = self.A_xy[level][s] != HierQV2._ILLEGAL
            else:
                mask[s] = 0  # Do nothing is illegal.
            if self.S_legal is not None:
                pass  # TODO Illegal move masking.

        # Sample an action (with preference for the end-goal if goal-conditioned policy).
        action = rand_argmax(self.critics[level].table[s, gc] * greedy, preference=pref, mask=mask)

        if level and self.relative_actions:  # Neighborhood-to-Absolute for sampled action.
            action = self.convert_action(level, reference=s, displaced=action, to_absolute=True)

        return action

    def update_critics(self) -> None:
        # At each level, update Q-tables for each trailing state for all possible subgoals.
        s_target = self.trace.raw[-1].next_state
        self.update_table(0, self.trace.raw[-1].state, self.trace.raw[-1].action, s_target)
        for i in range(1, self.n_levels):
            for h in reversed(range(self.trace.window(i))):
                self.update_table(i, self.trace[-(h + 1)].state, s_target, s_target, end_pos=self.trace[-1].goal)

    def update_table(self, level: int, s: int, a: int, s_next: int, end_pos: int = None) -> None:
        """Update Q-Tables with given transition """
        if level and self.relative_actions:  # Convert goal-action 'a' from Absolute-to-Neighborhood.
            a = self.convert_action(level, reference=s, displaced=s_next)

        # Hindsight action transition mask.
        if self.relative_goals:
            delta = np.asarray((self.S_xy[s], self.S_xy[s_next])).T
            g_her = self.ravel_delta_indices(np.diff(delta).T, r=self.atomic_horizons[level], motion=self.motion).item()

            goals = self.G[level]  # TODO: adjust goal-index for each trailing state.
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

    def update(self, _env: gym.Env, level: int, state: int, goal_stack: typing.List[int]) -> typing.Tuple[int, bool]:
        """Train level function of Algorithm 2 HierQ by (Levy et al., 2019).
        """
        step, done = 0, False
        while (step < self.horizons[level]) and (state not in goal_stack) and (not done):
            # Sample an action at the current hierarchy level.
            a = self.get_level_action(s=state, g=goal_stack[-1], level=level)

            if level:  # Temporally extend action as a goal through recursion.
                s_next, done = self.update(_env=_env, level=level - 1, state=state, goal_stack=goal_stack + [a])
            else:  # Atomic level: Perform a step in the actual environment, and perform critic-table updates.
                next_obs, r, done, meta = _env.step(a)
                s_next, terminal = self._get_index(meta['coord_next']), meta['goal_achieved']

                # Keep a memory of observed states to update Q-tables with hindsight.
                self.trace.add(GoalTransition(
                    state=state, goal=goal_stack[0], action=a, next_state=s_next, terminal=terminal, reward=None))

                # Update critic/ Q-tables given the current (global) trace of experience.
                self.update_critics()

            # Update state of control. Terminate level if new state is out of reach of current goal.
            state = s_next
            step += 1

        return state, done

    def clear_training_variables(self) -> None:
        # Clear all trailing states in each policy's deque.
        self.trace.reset()

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

            # Reset memory of training episode.
            self.clear_training_variables()

            done = False
            while (not done) and (state != goal_stack[0]):
                # Sample a goal as a temporally extended action and observe UMDP transition until env termination.
                state, done = self.update(
                    _env=_env, level=self.n_levels - 1, state=state, goal_stack=goal_stack)

            # Cleanup environment variables
            _env.close()
