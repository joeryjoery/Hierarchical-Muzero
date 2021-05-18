"""
This file implements a multi-step Hierarchical Q-learning algorithm generalized that subsumes HierQ by
(Levy et al., 2019) as a single-step instance.

Multi-step returns are computed using the Tree-Backup algorithm with a *greedy*-policy (Sutton and Barto, 2018).
"""
from __future__ import annotations
import typing
import sys

import numpy as np
import gym
import tqdm

from ..interface import Agent
from ..HierQ import TabularHierarchicalAgent

from .utils import CriticTable, GoalTransition, HierarchicalTrace

from mazelab_experimenter.utils import find, MazeObjects, rand_argmax, get_pos
from mazelab_experimenter.utils import ravel_moore_index, ravel_neumann_index, unravel_moore_index, \
    unravel_neumann_index, manhattan_distance, chebyshev_distance, neumann_neighborhood_size, moore_neighborhood_size


class HierQN(TabularHierarchicalAgent):
    _ILLEGAL: int = -1

    def __init__(self, observation_shape: typing.Tuple, n_actions: int, n_levels: int,
                 horizons: typing.Union[typing.List[int]], lr: float = 0.5, epsilon: float = 0.1,
                 discount: float = 0.95, n_steps: int = 1, relative_goals: bool = True,
                 universal_top: bool = True, legal_states: np.ndarray = None) -> None:
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

        if relative_goals:
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

                    states = HierQN._ILLEGAL * np.ones(len(coords))
                    states[mask] = self._get_index(coords[mask].T)

                    Ai_to_S.append(states.astype(np.int32))
                self.A_to_S.append(Ai_to_S)

            # With multiple hierarchies, the action horizon grows super exponentially.
            # If the subgoal-radius starts to exceed the environment dimensions, it is better to use absolute goals.
            if np.sum(np.asarray(self.G_radii) == max(self.observation_shape) // 2 - 1) > 1:
                print("Warning: multiple action spaces exceed the environment' dimensions, perhaps use absolute goals?")

        if not universal_top:
            # Do not train a goal-conditioned top-level policy (bit more memory efficient).
            self.G[-1] = [0]

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

            critic = CriticTable(init, dims, goal_conditioned=(i < self.n_levels - 1 or self.universal_top))
            critic.reset()

            self.critics.append(critic)

        # Data structures for keeping track of an environment trace along with a deque that retains previous
        # states within the hierarchical policies' action horizon. Last atomic_horizon value should be large.
        self.atomic_horizons = [int(np.prod(self.horizons[:i])) for i in range(0, self.n_levels + 1)]
        self.trace = HierarchicalTrace(num_levels=self.n_levels, horizons=self.atomic_horizons)
        self.trace.reset()

    @staticmethod
    def ravel_delta_indices(center_deltas: np.ndarray, r: int, motion: int) -> np.ndarray:
        if motion == Agent._MOORE_MOTION:
            return ravel_moore_index(center_deltas, radius=r, delta=True)
        else:
            return ravel_neumann_index(center_deltas, radius=r, delta=True)

    def convert_action(self, level: int, reference: int, displaced: int, to_absolute: bool = False) -> int:
        if to_absolute:
            return self.A_to_S[level][reference][displaced]
        else:
            return self.ravel_delta_indices((self.S_xy[displaced] - self.S_xy[reference])[None, :],
                                            r=self.G_radii[level - 1], motion=self.motion).item()

    def inside_radius(self, a: np.ndarray, b: np.ndarray, r: int) -> bool:
        """ Check whether the given arrays 'a' and 'b' are contained within the radius dependent on the motion. """
        return (manhattan_distance(a, b) if self.motion == Agent._NEUMANN_MOTION else chebyshev_distance(a, b)) < r

    def _get_index(self, coord: typing.Tuple, dims: typing.Optional[typing.Tuple[int, int]] = None) -> np.ndarray:
        return np.ravel_multi_index(coord, dims=self.observation_shape if dims is None else dims)

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
            if self.inside_radius(self.S_xy[s], self.G_xy[level][g], r=self.G_radii[level - 1]):
                pref = self.ravel_delta_indices(center_deltas=np.diff([self.S_xy[s], self.G_xy[level][g]]).T,
                                                r=self.G_radii[level - 1], motion=self.motion).item()

        mask = np.ones_like(self.critics[level].table[s, gc], dtype=bool)
        if level:  # Prune/ correct action space by masking out-of-bound or illegal subgoals.
            if self.relative_goals:
                mask = (self.A_to_S[level][s] != HierQN._ILLEGAL)
            else:
                mask[s] = 0  # Do nothing is illegal.
            if self.S_legal is not None:
                pass  # TODO Illegal move masking.

        # Sample an action (with preference for the end-goal if goal-conditioned policy).
        action = rand_argmax(self.critics[level].table[s, gc] * greedy, preference=pref, mask=mask)

        if level and self.relative_goals:  # Neighborhood-to-Absolute for sampled action.
            action = self.convert_action(level, reference=s, displaced=action, to_absolute=True)

        return action

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
                self.trace.add(GoalTransition(
                    state=state, goal=goal_stack[0], action=a, next_state=s_next, terminal=terminal, reward=None))

                if self.trace.raw[-1].degenerate:  # -> Only do a 1-step atomic update for illegal actions.
                    (s, g, a) = (state, self.G[0], a)
                    y = self.discount * self.critics[0].table[s, g].max(axis=-1)  # TODO: Init vs. max_a Q(s, g, a)
                    self.critics[0].table[s, g, a] += self.lr * (y - self.critics[0].table[s, g, a])
                else:
                    # Update every hierarchy level with a n-step Tree Backup.
                    for i in range(self.n_levels):
                        h = self.atomic_horizons[i]  # Helper variable for agent horizon.

                        # Look n * h steps back into transition data, sweep n to 1 if s_next is terminal.
                        for n in range(1 + (self.n_steps[i] - 1) * int(not terminal), 1 + self.n_steps[i]):
                            tau = len(self.trace) - h * (n - 1) - 1  # Inclusive latest time point inside window.
                            if tau >= 0:
                                # Compute shared (bootstrap) targets by every h'th atomic transition (for n steps).
                                ys = self.compute_target(i, self.trace[tau::h])
                                y_mask = np.flatnonzero(ys != 0.0)

                                # Extract the trailing window, being the h most recent transitions n*h steps back.
                                window = self.trace[:(tau + 1)][-h:]

                                # Update current Q values using computed targets and the trailing (s, a) pairs.
                                a_i = window[-1].next_state if i else window[-1].action  # shared action identifier.
                                for t in window:
                                    s_t, a_t = t.state, a_i  # Extract (s, a) pair for update.
                                    if i and self.relative_goals:  # Action correction for relative action spaces.
                                        a_t = self.convert_action(level=i, reference=t.state, displaced=a_i)

                                    # Perform Q-update over all goals simultaneously.
                                    self.critics[i].table[s_t, y_mask, a_t] += self.lr * (
                                            ys[y_mask] - self.critics[i].table[s_t, y_mask, a_t])

            # Update state of control.
            state = s_next
            step += 1

        return state, done

    def compute_target(self, level: int, target_transitions: typing.Sequence[GoalTransition]) -> np.ndarray:
        # Extract tail state.
        s_T, env_goal = target_transitions[-1].next_state, target_transitions[-1].goal

        # Bootstrap and reward mask at tail transition.
        r_T = r_mask = (s_T == self.G[level]) if self.critics[level].goal_conditioned else (s_T == env_goal)
        Q_T = self.critics[level].table[s_T, self.G[level]].max(axis=-1)

        # Compute tail target, and perform a n-step Tree Backup (if n > 1) using the greedy policy.
        G = r_T + (1 - r_mask) * self.discount * Q_T
        for k in reversed(range(len(target_transitions) - 1)):
            # Extract (s_k, a_k_next) pair. Abbrev. (s_k, a)
            s_k = target_transitions[k].next_state
            a = target_transitions[k+1].next_state if level else target_transitions[k+1].action

            if level and self.relative_goals:  # Action correction for relative action spaces.
                a = self.convert_action(level, reference=s_k, displaced=a)

            # Hindsight targets at every backup step. Line not reachable for non goal conditioned policy.
            r_k = r_mask = (s_k == self.G[level]) if self.critics[level].goal_conditioned else 0

            # Check whether the previously backed up target follows a greedy path. Reset trace if not.
            Q_k = self.critics[level].table[s_k, self.G[level]].max(axis=-1)
            G_mask = np.isclose(Q_k, self.critics[level].table[s_k, self.G[level], a])

            # Backup target.
            G = r_k + (1 - r_mask) * self.discount * ((1 - G_mask) * Q_k + G_mask * G)

        if self.critics[level].goal_conditioned:
            G[target_transitions[0].state] = self.critics[level].init  # Don't update for loops.

        return G

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
