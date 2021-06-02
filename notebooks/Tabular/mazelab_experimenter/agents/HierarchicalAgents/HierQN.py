"""
This file implements a multi-step Hierarchical Q-learning algorithm generalized that subsumes HierQ by
(Levy et al., 2019) as a single-step instance.

Multi-step returns are computed using the Tree-Backup algorithm with a *greedy*-policy (Sutton and Barto, 2018).
"""
from __future__ import annotations
import typing

import numpy as np
from .HierQV2 import HierQV2

from .utils import GoalTransition


class HierQN(HierQV2):

    def __init__(self, observation_shape: typing.Tuple, n_actions: int, n_levels: int,
                 horizons: typing.Union[typing.List[int]], lr: float = 0.5, epsilon: float = 0.1,
                 discount: float = 0.95, n_steps: int = 1,
                 relative_actions: bool = False, relative_goals: bool = False, universal_top: bool = False,
                 shortest_path_rewards: bool = False, sarsa: bool = False, stationary_filtering: bool = True,
                 hindsight_targets: bool = True, hindsight_goals: bool = True,
                 legal_states: np.ndarray = None, **kwargs) -> None:
        super().__init__(
            observation_shape=observation_shape, n_actions=n_actions, n_levels=n_levels,
            horizons=horizons, lr=lr, epsilon=epsilon, discount=discount,
            relative_actions=relative_actions, relative_goals=relative_goals, universal_top=universal_top,
            hindsight_targets=hindsight_targets, hindsight_goals=hindsight_goals,
            stationary_filtering=stationary_filtering, shortest_path_rewards=shortest_path_rewards, sarsa=sarsa,
            legal_states=legal_states, **kwargs
        )
        # Number of steps for backing up Q-targets.
        self.n_steps = np.full(n_levels, n_steps, dtype=np.int32) \
            if type(n_steps) == int else np.asarray(n_steps, dtype=np.int32)

    def update_table(self, level: int, horizon: int, trace: typing.Sequence[GoalTransition], **kwargs) -> None:
        # Look n * h steps back into transition data, sweep n to 1 if s_next is terminal.
        h = self.atomic_horizons[level]
        for n in range(1 + (self.n_steps[level] - 1) * int(not trace[-1].terminal), 1 + self.n_steps[level]):
            tau = len(trace) - h * (n - 1) - 1  # Inclusive latest time point inside window.
            if tau >= 0:
                # Extract the trailing window, being the h most recent transitions n*h steps back.
                window = trace[:(tau + 1)][-h:]

                # Compute shared (bootstrap) targets by every h'th atomic transition (for n steps).
                ys = self.compute_target(level, trace[tau::h])

                # Update current Q values using computed targets and the trailing (s, a) pairs.
                a_i = window[-1].next_state if level else window[-1].action  # shared action identifier.
                for t in window:
                    s_t, a_t = t.state, a_i  # Extract (s, a) pair for update.
                    if level and self.relative_actions:  # Action correction for relative action spaces.
                        a_t = self.convert_action(level=level, reference=t.state, displaced=a_i)

                    # Perform Q-update over all goals simultaneously for nonzero updates.
                    delta = ys - self.critics[level].table[s_t, self.G[level], a_t]
                    if self.relative_goals or (not self.hindsight_goals):
                        if self.critics[level].goal_conditioned:
                            # Bounded goals: Only update goal-table for the in-range goals
                            delta[self.goal_mask[level][s_t]] = 0

                    delta_mask = (delta != 0.0)
                    self.critics[level].table[s_t, delta_mask, a_t] += self.lr * delta[delta_mask]

    def compute_target(self, level: int, target_transitions: typing.Sequence[GoalTransition]) -> np.ndarray:
        """
        Uses the framework for tree-backup. Computes a n-step Q-learning target or a n-step expected sarsa target.
        """
        # Extract tail state.
        s_T, env_goal = target_transitions[-1].next_state, target_transitions[-1].goal[0]

        # Bootstrap and reward mask at tail transition.
        r_mask = (s_T == self.G[level]) if self.critics[level].goal_conditioned else (s_T == env_goal)
        Q_T = self.critics[level].table[s_T, self.G[level]].max(axis=-1)

        if self.sarsa:  # Expected SARSA target.
            p = self.epsilon[level]
            Q_T = p * (self.critics[level].table[s_T, self.G[level]]).mean(axis=-1) + (1 - p) * Q_T

        # Compute tail target, and perform a n-step Tree Backup (if n > 1) using the greedy policy.
        G = self.reward_func(r_mask, Q_T)
        for k in reversed(range(len(target_transitions) - 1)):
            # Extract (s_k, a_k_next) pair. Abbrev. (s_k, a)
            s_k = target_transitions[k].next_state
            a = target_transitions[k+1].next_state if level else target_transitions[k+1].action

            if level and self.relative_actions:  # Action correction for relative action spaces.
                a = self.convert_action(level, reference=s_k, displaced=a)

            # Hindsight targets at every backup step. Line not reachable for non goal conditioned policy.
            r_mask = (s_k == self.G[level]) if self.critics[level].goal_conditioned else 0

            # Check whether the previously backed up target is greedy. If not using sarsa, reset trace if non-greedy.
            Q_k = self.critics[level].table[s_k, self.G[level]].max(axis=-1)
            G_mask = (Q_k == self.critics[level].table[s_k, self.G[level], a]) | self.sarsa

            # Backup target.
            bootstrap_k = ((1 - G_mask) * Q_k + G_mask * G)
            G = self.reward_func(r_mask, bootstrap_k)

        if self.critics[level].goal_conditioned:
            G[target_transitions[0].state] = self.critics[level].init  # Don't update transition loops

        return G
