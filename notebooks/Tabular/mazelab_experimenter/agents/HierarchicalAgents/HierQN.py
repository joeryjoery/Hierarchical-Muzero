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
                 discount: float = 0.95, n_steps: int = 1, relative_goals: bool = False, relative_actions: bool = True,
                 universal_top: bool = True, legal_states: np.ndarray = None, **kwargs) -> None:
        super().__init__(
            observation_shape=observation_shape, n_actions=n_actions, n_levels=n_levels,
            horizons=horizons, lr=lr, epsilon=epsilon, discount=discount,
            relative_actions=relative_actions, relative_goals=relative_goals, universal_top=universal_top,
            shortest_path_rewards=False, legal_states=legal_states, **kwargs
        )
        # Number of steps for backing up Q-targets.
        self.n_steps = np.full(n_levels, n_steps, dtype=np.int32) \
            if type(n_steps) == int else np.asarray(n_steps, dtype=np.int32)

    def update_critics(self) -> None:   # TODO: functionality relative goals.
        # Update every policy with a n-step "Tree Backup" (n-step Q-learning).
        for i in range((self.n_levels if not self.trace.raw[-1].degenerate else 1)):
            # Always update atomic policy under the raw trace, only update higher levels on actual transitions.
            trace = self.trace.transitions if i else self.trace.raw
            h = self.atomic_horizons[i]  # Helper variable for agent horizon.

            # Look n * h steps back into transition data, sweep n to 1 if s_next is terminal.
            for n in range(1 + (self.n_steps[i] - 1) * int(not trace[-1].terminal), 1 + self.n_steps[i]):
                tau = len(trace) - h * (n - 1) - 1  # Inclusive latest time point inside window.
                if tau >= 0:
                    # Compute shared (bootstrap) targets by every h'th atomic transition (for n steps).
                    ys = self.compute_target(i, trace[tau::h])

                    # Extract the trailing window, being the h most recent transitions n*h steps back.
                    window = trace[:(tau + 1)][-h:]

                    # Update current Q values using computed targets and the trailing (s, a) pairs.
                    a_i = window[-1].next_state if i else window[-1].action  # shared action identifier.
                    for t in window:
                        s_t, a_t = t.state, a_i  # Extract (s, a) pair for update.
                        if i and self.relative_actions:  # Action correction for relative action spaces.
                            a_t = self.convert_action(level=i, reference=t.state, displaced=a_i)

                        # Perform Q-update over all goals simultaneously for nonzero updates.
                        delta = ys - self.critics[i].table[s_t, self.G[i], a_t]
                        delta_mask = np.flatnonzero(delta != 0.0)
                        self.critics[i].table[s_t, delta_mask, a_t] += self.lr * delta[delta_mask]

    # TODO: functionality alternative reward functions.
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

            if level and self.relative_actions:  # Action correction for relative action spaces.
                a = self.convert_action(level, reference=s_k, displaced=a)

            # Hindsight targets at every backup step. Line not reachable for non goal conditioned policy.
            r_k = r_mask = (s_k == self.G[level]) if self.critics[level].goal_conditioned else 0

            # Check whether the previously backed up target follows a greedy path. Reset trace if not.
            Q_k = self.critics[level].table[s_k, self.G[level]].max(axis=-1)
            G_mask = (Q_k == self.critics[level].table[s_k, self.G[level], a])

            # Backup target.
            G = r_k + (1 - r_mask) * self.discount * ((1 - G_mask) * Q_k + G_mask * G)

        if self.critics[level].goal_conditioned:
            G[target_transitions[0].state] = 0.0  # Don't update transition loops

        return G
