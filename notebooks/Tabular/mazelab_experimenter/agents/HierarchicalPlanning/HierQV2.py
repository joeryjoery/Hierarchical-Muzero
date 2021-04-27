from __future__ import annotations
import typing
import sys
from dataclasses import dataclass, field
from collections import deque

import numpy as np
import gym
import tqdm

from ..interface import Agent
from ..HierQ import TabularHierarchicalAgent

from mazelab_experimenter.utils import find, MazeObjects, rand_argmax
from mazelab_experimenter.utils import ravel_moore_index, ravel_neumann_index, unravel_moore_index, \
    unravel_neumann_index, manhattan_distance, chebyshev_distance


class HierQV2(TabularHierarchicalAgent):

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
                 discount: float = 0.95, relative_goals: bool = True, universal_top: bool = True,
                 legal_states: np.ndarray = None) -> None:
        super().__init__(observation_shape=observation_shape, n_actions=n_actions, n_levels=n_levels)

        assert type(horizons) == int or (type(horizons) == list and len(horizons) == n_levels), \
            f"Incorrect specification of horizon lengths, " \
            f"this should either be a fixed integer or a list in ascending hierarchy order."

        # Initialize the agent's state space and goal space
        self.S, self.S_legal = np.arange(np.prod(observation_shape)), legal_states
        self.G = [self.S] * n_levels
        self.A = [np.arange(self.n_actions)] + self.G[:-1]  # Defaults (A \equiv G \equiv S for pi_i, i > 0).

        if relative_goals:
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
        self.epsilon = epsilon
        self.discount = discount
        self.relative_goals = relative_goals
        self.universal_top = universal_top
        self.horizons = np.full(n_levels, horizons, dtype=np.int32) \
            if type(horizons) == int else np.asarray(horizons, dtype=np.int32)

        # Initialize Q-tables at all levels pessimistically with value '-Horizon_i'.
        self.critics = list()
        for i in range(self.n_levels):
            init = -np.clip(np.prod(self.horizons[:(i + 1)]), 0, max(horizons))
            dims = (len(self.S), len(self.G[i]), len(self.A[i]))

            critic = HierQV2.CriticTable(init, dims, goal_conditioned=(i < self.n_levels - 1 or self.universal_top))
            critic.reset()

            self.critics.append(critic)

        # Previous states array keeps track of positions during training for each level i, 0 < i <= n_levels
        self._previous_states = [deque(maxlen=int(np.prod(self.horizons[:(i + 1)]))) for i in range(self.n_levels - 1)]
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
        return int(np.ravel_multi_index(coord, dims=self.observation_shape if dims is None else dims))

    def reset(self, full_reset: bool = False) -> None:
        self.clear_hierarchy(self.n_levels - 1 - int(not full_reset))
        for critic in self.critics:
            critic.reset()

    def update_critic(self, level: int, s: int, a: int, s_next: int, end_pos: int = None) -> None:
        """Update Q-Tables with given transition """

        if level and self.relative_goals:  # Convert subgoal-action 'a' from Absolute-to-Neighborhood.
            delta = np.asarray(np.unravel_index([s, a], self.observation_shape))
            a = self.ravel_delta_indices(np.diff(delta).T, r=self.G_radii[level - 1], motion=self.motion).item()

        goals = self.G[level]
        mask = goals != s_next          # Hindsight action transition mask.
        if not self.critics[level].goal_conditioned:
            goals = 0                   # Correct for non-universal critic.
            mask = end_pos != s_next    # Correct hindsight action transition mask.

        # Q-Learning update for each goal-state.
        ys = mask * (-1 + self.discount * np.amax(self.critics[level].table[s_next, goals], axis=-1))
        self.critics[level].table[s, goals, a] += self.lr * (ys - self.critics[level].table[s, goals, a])

    def get_level_action(self, s: int, g: int, level: int, explore: bool = True) -> int:
        """Sample a **Hierarchy action** (not an Environment action) from the Agent. """
        # Helper variables
        greedy = int(level or (not explore) or (np.random.random_sample() > self.epsilon))
        gc = g * int(self.critics[level].goal_conditioned)

        pref = g if level else None        # Goal-selection bias for shared max Q-values.
        if level and self.relative_goals:  # Action Preference: Absolute-to-Neighborhood for policy Action-Space
            pref = None
            if HierQV2.inside_radius(self.S_xy[s], self.G_xy[level][g], r=self.G_radii[level - 1], motion=self.motion):
                pref = self.ravel_delta_indices(center_deltas=np.diff([self.S_xy[s], self.G_xy[level][g]]).T,
                                                r=self.G_radii[level - 1], motion=self.motion).item()

        mask = np.ones_like(self.critics[level].table[s, gc], dtype=bool)
        if level:  # Prune/ correct action space by masking out-of-bound or illegal subgoals.
            if self.relative_goals:
                mask[len(mask) // 2] = 0  # Do nothing is illegal.

                c = self.S_xy[s] + self.A_dxy[level]  # Check all coordinates for out of bound cases.
                mask &= np.all((0, 0) <= c, axis=-1) & np.all(c < self.observation_shape, axis=-1)
            else:
                mask[s] = 0  # Do nothing is illegal.
            if self.S_legal is not None:
                pass  # TODO Illegal move masking.

        # Sample an action (with preference for the end-goal if goal-conditioned policy).
        action = rand_argmax(self.critics[level].table[s, gc] * greedy, preference=pref, mask=mask)

        if level and self.relative_goals:  # Neighborhood-to-Absolute for sampled action.
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
                if state != s_next:
                    self._transitions.append((state, a, s_next, ))
                    for d in self._previous_states:
                        d.append(state)

                # At each level, update Q-tables for each trailing state for all possible subgoals.
                # print(state, a, s_next)
                self.update_critic(0, state, a, s_next)
                for i in range(1, self.n_levels):
                    for s_mem in self._previous_states[i - 1]:
                        self.update_critic(i, s_mem, s_next, s_next, end_pos=goal_stack[0])

            # Update state of control.
            state = s_next
            step += 1

        return state, done

    def train(self, _env: gym.Env, num_episodes: int, progress_bar: bool = False, **kwargs) -> None:
        """Train the HierQ agent through recursion with the `self.update` function.
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


class HierQV2Indev(TabularHierarchicalAgent):

    @dataclass
    class GoalTransition:
        state: np.ndarray
        goal: typing.Union[int, np.ndarray]  # Can be intrinsic, extrinsic, a singular goal, or multiple (HER)
        action: int
        next_state: np.ndarray
        terminal: bool
        reward: typing.Optional[float]  # Rewards can be extrinsic or derived from (state == goal).

        @property
        def degenerate(self) -> bool:
            return self.state == self.next_state

    @dataclass
    class HierarchicalTrace:  # Accessible as a Sequence[GoalTransition] object.
        num_levels: int  # Number of hierarchies.
        horizons: typing.List  # Trace-window for each hierarchy level and the environment level.
        non_degenerate: bool = True  # Whether to access the trace using only the transitions where state != state_next
        full: typing.List[HierQV2Indev.GoalTransition] = field(default_factory=list)  # Raw unfiltered trace.
        transitions: typing.List[int] = field(default_factory=list)  # Trace indices where state != state_next

        def __len__(self) -> int:
            return len(self.transitions) if self.non_degenerate else len(self.full)

        def __getitem__(self, t: typing.Union[int, slice]) -> HierQV2Indev.GoalTransition:
            return [self.full[i] for i in [*self.transitions[t]]] if self.non_degenerate else self.full[t]

        def window(self, level: int) -> int:
            """Get the size of the currently open window of trailing states for the given hierarchy level."""
            return min([self.horizons[level], len(self)])

        def add(self, transition: HierQV2Indev.GoalTransition) -> None:
            self.full.append(transition)
            if not transition.degenerate:
                self.transitions.append(len(self.full) - 1)

        def reset(self):
            self.full.clear()
            self.transitions.clear()

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
                 discount: float = 0.95, n_steps: int = 1, relative_goals: bool = True, universal_top: bool = True,
                 legal_states: np.ndarray = None) -> None:
        super().__init__(observation_shape=observation_shape, n_actions=n_actions, n_levels=n_levels)

        assert type(horizons) == int or (type(horizons) == list and len(horizons) == n_levels), \
            f"Incorrect specification of horizon lengths, " \
            f"this should either be a fixed integer or a list in ascending hierarchy order."

        # Initialize the agent's state space and goal space
        self.S, self.S_legal = np.arange(np.prod(observation_shape)), legal_states
        self.G = [self.S] * n_levels
        self.A = [np.arange(self.n_actions)] + self.G[:-1]  # Defaults (A \equiv G \equiv S for pi_i, i > 0).

        if relative_goals:
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
        self.epsilon = epsilon
        self.discount = discount
        self.n_steps = np.full(n_levels, n_steps, dtype=np.int32) \
            if type(n_steps) == int else np.asarray(n_steps, dtype=np.int32)
        self.relative_goals = relative_goals
        self.universal_top = universal_top
        self.horizons = np.full(n_levels, horizons, dtype=np.int32) \
            if type(horizons) == int else np.asarray(horizons, dtype=np.int32)

        # Initialize Q-tables at all levels pessimistically with value '-Horizon_i'.
        self.critics = list()
        for i in range(self.n_levels):
            init = 0.0
            dims = (len(self.S), len(self.G[i]), len(self.A[i]))

            critic = HierQV2.CriticTable(init, dims, goal_conditioned=(i < self.n_levels - 1 or self.universal_top))
            critic.reset()

            self.critics.append(critic)

        # Data structures for keeping track of an environment trace along with a deque that retains previous
        # states within the hierarchical policies' action horizon. Last atomic_horizon value should be large.
        atomic_horizons = [int(np.prod(self.horizons[:i])) for i in range(0, self.n_levels + 1)]
        self.trace = HierQV2Indev.HierarchicalTrace(num_levels=self.n_levels, horizons=atomic_horizons)
        self.trace.reset()

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
        return int(np.ravel_multi_index(coord, dims=self.observation_shape if dims is None else dims))

    def reset(self, full_reset: bool = False) -> None:
        self.clear_hierarchy(self.n_levels - 1 - int(not full_reset))
        for critic in self.critics:
            critic.reset()

    def get_level_action(self, s: int, g: int, level: int, explore: bool = True) -> int:
        """Sample a **Hierarchy action** (not an Environment action) from the Agent. """
        # Helper variables
        greedy = int(level or (not explore) or (np.random.random_sample() > self.epsilon))
        gc = g * int(self.critics[level].goal_conditioned)

        pref = g if level else None        # Goal-selection bias for shared max Q-values.
        if level and self.relative_goals:  # Action Preference: Absolute-to-Neighborhood for policy Action-Space
            pref = None
            if HierQV2.inside_radius(self.S_xy[s], self.G_xy[level][g], r=self.G_radii[level - 1], motion=self.motion):
                pref = self.ravel_delta_indices(center_deltas=np.diff([self.S_xy[s], self.G_xy[level][g]]).T,
                                                r=self.G_radii[level - 1], motion=self.motion).item()

        mask = np.ones_like(self.critics[level].table[s, gc], dtype=bool)
        if level:  # Prune/ correct action space by masking out-of-bound or illegal subgoals.
            if self.relative_goals:
                mask[len(mask) // 2] = 0  # Do nothing is illegal.

                c = self.S_xy[s] + self.A_dxy[level]  # Check all coordinates for out of bound cases.
                mask &= np.all((0, 0) <= c, axis=-1) & np.all(c < self.observation_shape, axis=-1)
            else:
                mask[s] = 0  # Do nothing is illegal.
            if self.S_legal is not None:
                pass  # TODO Illegal move masking.

        # Sample an action (with preference for the end-goal if goal-conditioned policy).
        action = rand_argmax(self.critics[level].table[s, gc] * greedy, preference=pref, mask=mask)

        if level and self.relative_goals:  # Neighborhood-to-Absolute for sampled action.
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
                s_next, terminal = self._get_index(meta['coord_next']), meta['goal_achieved']

                # Keep a memory of observed states to update Q-tables with hindsight.
                self.trace.add(HierQV2Indev.GoalTransition(
                    state=state, goal=goal_stack[0], action=a, next_state=s_next, terminal=terminal, reward=None))

                if self.trace.full[-1].degenerate:  # -> Only do a 1-step atomic update for illegal actions.
                    self.tree_backup(0, *self.build_backup_tree(self.trace.full, step=0, horizon=1, n=1))
                else:
                    # Update every hierarchy level with a n-step Tree Backup.
                    for i in range(self.n_levels):
                        # Update critics based on the trailing states within the agent window.  TODO: Batch-update.
                        for h in range(self.trace.window(i)):
                            # Construct n-step backup tree. Uses n = 0, ..., N at trace termination.
                            for n in range(1 + (self.n_steps[i] - 1) * int(not terminal), 1 + self.n_steps[i]):
                                if h + self.trace.horizons[i] * (n - 1) < len(self.trace):
                                    self.tree_backup(i, *self.build_backup_tree(
                                        self.trace, step=h, horizon=self.trace.horizons[i], n=n))

            # Update state of control.
            state = s_next
            step += 1

        return state, done

    @staticmethod
    def build_backup_tree(trace: typing.Sequence[HierQV2Indev.GoalTransition],
                          step: int, horizon: int, n: int) -> typing.Tuple[typing.List[HierQV2Indev.GoalTransition],
                                                                           typing.List[HierQV2Indev.GoalTransition]]:
        return trace[-(step + 1 + horizon * (n - 1))::horizon], trace[-(1 + horizon * (n - 1))::horizon]

    def tree_backup(self, level: int, base_transitions: typing.List[HierQV2Indev.GoalTransition],
                    target_transitions: typing.List[HierQV2Indev.GoalTransition] = None) -> None:
        # Basic n-step Tree Backup based on the given transition list (time-index invariant).
        # > t-start and t-next serve as the (state, action) pair and t_end serves as the n-step lookahead target.
        t_start, t_next, t_end = base_transitions[0], target_transitions[0], target_transitions[-1]

        # Get state-action for the Q-update.
        s = t_start.state
        a = t_next.next_state if level else t_start.action
        if level and self.relative_goals:  # Action correction for relative action spaces.
            delta = np.asarray(np.unravel_index([s, a], self.observation_shape))
            a = self.ravel_delta_indices(np.diff(delta).T, r=self.G_radii[level - 1], motion=self.motion).item()

        # Terminal reward calculation based on achieving state-goals.
        goals, mask = (self.G[level], self.G[level] == t_end.next_state) \
            if self.critics[level].goal_conditioned else (0, t_end.terminal)

        # Target calculation
        r_T = mask
        Q_T = np.amax(self.critics[level].table[t_end.next_state, goals], axis=-1)
        G = r_T + (1 - mask) * self.discount * Q_T   # n-step lookahead target.

        # Backup from leaf node, cut traces where a != a* and reset G to a new bootstrap target.
        for k in reversed(range(0, len(base_transitions) - 1)):
            t_k, t_k_next = base_transitions[k], target_transitions[k]  # For atomic policy t_k == t_k_next

            # Extract (s, a) pair for backup.
            s_k, a_k = t_k.state, (t_k_next.next_state if level else t_k.action)

            if level and self.relative_goals:   # Action correction for relative action spaces.
                delta = np.asarray(np.unravel_index([s_k, a_k], self.observation_shape))
                a_k = self.ravel_delta_indices(np.diff(delta).T, r=self.G_radii[level - 1], motion=self.motion).item()

            # Hindsight targets at every backup step. t_k_next.terminal should always be 0.
            r_mask = goals == t_k_next.next_state if self.critics[level].goal_conditioned else t_k_next.terminal

            # Reset trace if selected action isn't Q-optimal (don't reset in case of ties).
            Q_k = np.max(self.critics[level].table[s_k, goals], axis=-1)
            r_k = r_mask

            # Backup G if a_k is a greedy action AND the transition is not terminal in hindsight.
            G_mask = np.isclose(Q_k, self.critics[level].table[s_k, goals, a_k])  # boolean array
            G = r_k + self.discount * (1 - r_mask) * ((1 - G_mask) * Q_k + G_mask * G)

        # n-step SARSA update
        self.critics[level].table[s, goals, a] += self.lr * (G - self.critics[level].table[s, goals, a])

    def train(self, _env: gym.Env, num_episodes: int, progress_bar: bool = False, **kwargs) -> None:
        """Train the HierQ agent through recursion with the `self.update` function.
        """
        iterator = (tqdm.trange(num_episodes, file=sys.stdout, desc="HierQ Training")
                    if progress_bar else range(num_episodes))

        for _ in iterator:
            # Reinitialize environment after each episode.
            state = self._get_index(self._get_pos(_env.reset()))

            # Set top hierarchy goal/ environment goal (only one goal-state is supported for now)
            goal_stack = [self._get_index(_env.unwrapped.maze.get_end_pos()[0])]

            # Refresh trace-data.
            self.trace.reset()

            done = False
            while (not done) and (state != goal_stack[0]):
                # Sample a goal as a temporally extended action and observe UMDP transition until env termination.
                a = self.get_level_action(s=state, g=goal_stack[-1], level=self.n_levels - 1)

                state, done = self.update(
                    _env=_env, level=self.n_levels - 2, state=state, goal_stack=goal_stack + [a])

            # Cleanup environment variables
            _env.close()
