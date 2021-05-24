from __future__ import annotations
import typing

import numpy as np
from .HierQV2 import HierQV2

from .utils import GoalTransition, HierarchicalEligibilityTrace


class HierQLambda(HierQV2):
    # TODO: relative goal functionality and alternative reward functions.
    #  PLUS Eligiblity trace data structure rework

    def __init__(self, observation_shape: typing.Tuple, n_actions: int, n_levels: int,
                 horizons: typing.Union[typing.List[int]], lr: float = 0.5, epsilon: float = 0.1,
                 discount: float = 0.95, decay: float = 0.9, relative_goals: bool = False,
                 relative_actions: bool = True, universal_top: bool = True,
                 legal_states: np.ndarray = None, **kwargs) -> None:
        super().__init__(
            observation_shape=observation_shape, n_actions=n_actions, n_levels=n_levels,
            horizons=horizons, lr=lr, epsilon=epsilon, discount=discount,
            relative_actions=relative_actions, relative_goals=relative_goals, universal_top=universal_top,
            shortest_path_rewards=False, legal_states=legal_states, **kwargs
        )
        # Lambda value for decaying compound returns inside the eligibility trace.
        self.decay = decay

        # Initialize for each hierarchy level its individual Hierarchical-Eligibility trace.
        self.eligibility = list()
        for i in range(self.n_levels):
            HTrace = HierarchicalEligibilityTrace(self.atomic_horizons[i], len(self.G[i]))
            HTrace.reset()

            self.eligibility.append(HTrace)

    def clear_training_variables(self) -> None:
        """ Additionally reset eligibility traces along with parent functionality. """
        super().clear_training_variables()
        self.trace.reset()
        for trace in self.eligibility:
            trace.reset()

    def update_critics(self) -> None:
        # Always update atomic policy under the raw trace, update higher level policies only on actual transitions.
        self.update_critic(level=0, horizon=1, trace=self.trace.raw)
        if not self.trace.raw[-1].degenerate:
            # Update Q tables at each level using built trace.
            for i in range(1, self.n_levels):
                self.update_critic(i, self.trace.window(i), self.trace.transitions)

    def update_critic(self, level: int, horizon: int, trace: typing.Sequence[GoalTransition]) -> None:
        # Helper variables
        h_i = (len(trace) - 1) % horizon
        window = trace[-horizon:]
        goals, env_goal = self.G[level], window[-1].goal

        # Extract most recent (s, a) pair inside the trace with maximum horizon.
        s_h, s_t = window[0].state, window[-1].next_state
        a_h = s_t if level else window[-1].action
        if level and self.relative_actions:
            a_h = self.convert_action(level, reference=s_h, displaced=s_t)

        # 1-step return values.
        r_t = r_mask = (s_t == goals) if self.critics[level].goal_conditioned else (s_t == env_goal)
        G = r_t + (1 - r_mask) * self.discount * self.critics[level].table[s_t, goals].max(axis=-1)

        ### BEGIN Compound Update for the Maximally Temporally Extended action == multi-step updates.
        # Decay eligibility trace and cut-traces for non-greedy actions and hindsight terminal states (per goal).
        Q_h = self.critics[level].table[s_h, goals].max(axis=-1)
        mask = (Q_h == self.critics[level].table[s_h, goals, a_h])

        if self.critics[level].goal_conditioned:
            mask[s_h] = False  # Cut trace behind the 'current-state' goal (--> already achieved).

        # Cut and decay trace accordingly and add the new maximal-horizon action.
        self.eligibility[level].cut(h=h_i, indices=np.flatnonzero(~mask), t=len(trace)-1)
        self.eligibility[level].add(h=h_i, s=s_h, a=a_h, t=len(trace)-1)
        # self.eligibility[level].trace[h_i][:, ~mask, :] = 0.0
        # self.eligibility[level].trace[h_i][:, mask, :] *= (self.decay * self.discount)
        # self.eligibility[level].trace[h_i][s_h, :, a_h] = 1.0

        # Compute Maximal horizon telescoping/ TD error.
        delta = G - self.critics[level].table[s_h, goals, a_h]
        goal_pool = np.flatnonzero(delta != 0)

        cut_points = self.eligibility[level].cuts[h_i][goal_pool]
        t_truncate = min(cut_points, default=len(trace))

        e = 1.0
        seen = set()
        entry = self.eligibility[level].trace[h_i][-1]
        for t in reversed(range(t_truncate, len(trace))):
            if (not entry) or (e < self.eligibility[level].TRUNCATE):
                break

            pool = goal_pool[cut_points <= t]
            while entry and (entry.time == t):
                # Replacing trace: update all (s, a) once per (masked) goal for the maximal eligibility.
                if (entry.state, entry.action) not in seen:
                    self.critics[level].table[entry.state, pool, entry.action] += self.lr * e * delta[pool]
                    seen.add((entry.state, entry.action))

                # Step backwards inside the trace
                entry = entry.previous

            # Decay eligibility value backward through time.
            e *= self.decay * self.discount

        # target = self.eligibility[level].trace[h_i][:, delta_mask, :] * (self.lr * delta[None, delta_mask, None])

        # Broadcast TD-error over goal dimension for the non-zero backups.
        # self.critics[level].table[:, delta_mask, :] += target

        ### END Compound Update

        ### Begin Trailing States Update == 1-step updates.
        for t in window[1:]:
            s, a = t.state, a_h  # Extract (s, a) pair for update.
            if level and self.relative_actions:  # Action correction for relative action spaces.
                a = self.convert_action(level, reference=t.state, displaced=s_t)

            # Basic 1-step Q-learning update for masked targets.
            delta = (G - self.critics[level].table[s, goals, a])
            delta_mask = (delta != 0)
            self.critics[level].table[s, delta_mask, a] += self.lr * delta[delta_mask]

            # Add trailing (s, a) pair to eligibility trace.
            self.eligibility[level].add(h=h_i, s=s, a=a, t=len(trace)-1)
            # self.eligibility[level].trace[h_i][s, :, a] = 1.0
