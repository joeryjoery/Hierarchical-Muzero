import typing
import sys
from collections import deque

import numpy as np
import gym
import tqdm

from mazelab_experimenter.agents.interface import Agent
from ..HierQ import TabularHierarchicalAgent

from mazelab_experimenter.utils import find, MazeObjects, rand_argmax
from mazelab_experimenter.utils import ravel_moore_index, ravel_neumann_index, unravel_moore_index, \
    unravel_neumann_index, manhattan_distance, chebyshev_distance


class HierQTD(TabularHierarchicalAgent):

    def __init__(self, observation_shape: typing.Tuple, n_actions: int, n_levels: int,
                 horizons: typing.Union[typing.List[int]], lr: float = 0.5, epsilon: float = 0.1,
                 discount: float = 0.95, universal_top: bool = True,
                 ignore_training_time: bool = False, legal_states: np.ndarray = None) -> None:
        super().__init__(observation_shape=observation_shape, n_actions=n_actions, n_levels=n_levels)

        assert type(horizons) == int or (type(horizons) == list and len(horizons) == n_levels), \
            f"Incorrect specification of horizon lengths, " \
            f"this should either be a fixed integer or a list in ascending hierarchy order."

        # Initialize the agent's state space, goal space, and action space.
        self.S, self.S_legal = np.arange(np.prod(observation_shape)), legal_states
        self.G = [self.S] * n_levels

        # Set goals to the tiles centered around each state with radius_i = horizon_i (TOP level remains absolute)
        self.G_radii = [np.prod(horizons[:(i + 1)]) for i in range(n_levels - 1)]
        self.A = [np.arange(self.n_actions)] + [np.arange(self._goal_tiles(r, self.motion)) for r in self.G_radii]

        # With multiple hierarchies, the action horizon grows super exponentially.
        # If the subgoal-radius starts to exceed the environment dimensions, it is better to use absolute goals.
        if np.sum(np.asarray(self.G_radii) == max(self.observation_shape) // 2 - 1) > 1:
            print("Warning: multiple action spaces exceed the environment' dimensions.")

        if not universal_top:
            # Do not train a goal-conditioned top-level policy (bit more memory efficient).
            self.G[-1] = [None]

        # Training and agent hyperparameters.
        self.lr = lr
        self.epsilon = epsilon
        self.discount = discount
        self.universal_top = universal_top
        self.ignore_training_time = ignore_training_time
        self.horizons = np.full(n_levels, horizons, dtype=np.int32) \
            if type(horizons) == int else np.asarray(horizons, dtype=np.int32)

        # Initialize critic tables at all levels pessimistically with value '-Horizon_i'.
        self._critic_init = -np.clip([np.prod(self.horizons[:(i + 1)]) for i in range(self.n_levels)], 0, max(horizons))
        self._critic_tables = [
            np.full((len(self.S), len(self.G[0]), len(self.A[0])), self._critic_init[0], dtype=np.float32)
        ] + [
            np.full((len(self.S), len(self.G[i])), self._critic_init[i], dtype=np.float32)
            for i in range(1, self.n_levels)
        ]
        eye = np.stack([self.S, self.S], axis=0).T
        for i in range(1, self.n_levels - int(not universal_top)):  # State = Goal = 0
            self._critic_tables[i][eye] = 0

        # Previous states array keeps track of positions during training for each level i, 0 < i <= n_levels
        self._previous_states = [deque(maxlen=int(np.prod(self.horizons[:(i + 1)]))) for i in range(self.n_levels - 1)]

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
        for table, value in zip(self._critic_tables, self._critic_init):
            table[...] = value

    def update_critic(self, level: int, s: int, a: int, s_next: int, end_pos: int = None) -> None:
        """Update Critic Tables using Q-learning update rule given the transition """
        if not level:  # Atomic level Q-learning update
            gs = self.G[level]
            mask = gs != s_next

            # Q-Learning update for each goal-state.
            ys = mask * (-1 + self.discount * np.amax(self._critic_tables[level][s_next, gs], axis=-1))
            self._critic_tables[level][s, gs, a] += self.lr * (ys - self._critic_tables[level][s, gs, a])
        else:  # Subgoal level Q-learning update on the state Value table.

            goals = self.G[level]
            mask = goals != s_next          # Hindsight action transition mask.
            if level == self.n_levels - 1 and not self.universal_top:
                goals = 0                   # Correct for non-universal critic.
                mask = end_pos != s_next    # Correct hindsight action transition mask.

            # Q-learning update at s_t and s_(t+1) for each goal-state.
            v = self._critic_tables[level]
            v[s, goals] += self.lr * (-1 + mask * self.discount * v[s_next, goals] - v[s, goals])

    def get_level_action(self, s: int, g: int, level: int, explore: bool = True) -> int:
        """Sample a **Hierarchy action** (not an Environment action) from the Agent. """
        greedy = int(level or not explore or np.random.random_sample() > self.epsilon)
        if not level:  # Atomic policy
            return rand_argmax(self._critic_tables[level][s][g] * greedy)
        # else: Subgoal sampling policies:

        coord_sg = np.asarray(np.unravel_index([s, g], self.observation_shape))
        coord_delta = np.diff(coord_sg).T

        pref = None  # Goal-selection bias for shared max Q-values.
        if level:    # Action Preference: Absolute-to-Neighborhood for policy Action-Space
            if HierQTD.inside_radius(coord_sg[:, 0], coord_sg[:, 1], r=self.G_radii[level - 1], motion=self.motion):
                pref = self.ravel_delta_indices(
                    center_deltas=coord_delta, r=self.G_radii[level - 1], motion=self.motion).item()

        # Map action-pace to state-space indices for the critic table lookup. Clip illegal actions (mask later).
        coord_deltas = self.unravel_delta_indices(self.A[level], self.G_radii[level - 1], self.motion)
        coords = coord_sg[:, 0] + coord_deltas
        forward_states = np.ravel_multi_index(coords.T, dims=self.observation_shape, mode='clip')

        # Prune/ correct action space by masking out-of-bound or illegal subgoals.
        mask = np.ones_like(self.A[level], dtype=bool)
        mask &= np.all((0, 0) <= coords, axis=-1) & np.all(coords < self.observation_shape, axis=-1)  # Out of bounds.
        if self.S_legal is not None:
            pass  # TODO Illegal move masking.

        # Correct Goal-Conditioning index if top level isn't universal.
        gc = g * int(level != self.n_levels - 1 or self.universal_top)

        # Use state-space coordinates to perform 1-ply Breadth-First critic table lookup and sample a subgoal.
        vs = self._critic_tables[level][forward_states, gc]
        action = self.A[level][rand_argmax(vs * greedy, preference=pref, mask=mask)]

        # Neighborhood-to-Absolute for sampled action.
        coord_delta = self.unravel_delta_indices(np.asarray([action]), self.G_radii[level - 1], self.motion)
        coord_action = coord_sg[:, 0] + coord_delta
        action_state = self._get_index(coord_action.ravel())

        return action_state

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
        while step < self.horizons[level] and state not in goal_stack and (not done or self.ignore_training_time):
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
                        d.append(s_next)

                # At each level, update Q-tables for each trailing state for all possible subgoals.
                # print(state, a, s_next)
                self.update_critic(0, state, a, s_next)
                for i in range(1, self.n_levels):
                    for s_mem in self._previous_states[i - 1]:
                        self.update_critic(i, s_mem, s_next, s_next, end_pos=goal_stack[0])

            # Update state of control.
            state = s_next

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
            while not done and state != goal_stack[0]:
                # Sample a goal as a temporally extended action and observe UMDP transition until env termination.
                a = self.get_level_action(s=state, g=goal_stack[-1], level=self.n_levels - 1)
                state, done = self.update(
                    _env=_env, level=self.n_levels - 2, state=state, goal_stack=goal_stack + [a])

            # Cleanup environment variables
            _env.close()
