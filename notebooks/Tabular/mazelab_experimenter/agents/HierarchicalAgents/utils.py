from __future__ import annotations
from dataclasses import dataclass, field
import typing

import numpy as np


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
class HierarchicalTrace:
    """ Data structure for manipulation convenience.

    Access an agent's environment trace as a Sequence[GoalTransition] object.

    """
    num_levels: int             # Number of hierarchies.
    horizons: typing.List[int]  # Trace-window for each hierarchy level and the environment level.
    raw: typing.List[GoalTransition] = field(default_factory=list)          # Raw unfiltered trace.
    transitions: typing.List[GoalTransition] = field(default_factory=list)  # >> state != state_next

    def __len__(self) -> int:
        """ Length of the non-degenerate transitions of the trace. """
        return len(self.transitions)

    def __getitem__(self, t: typing.Union[int, slice]) -> typing.Union[typing.Sequence[GoalTransition], GoalTransition]:
        """ Access the non-degenerate transitions of the trace. """
        return self.transitions[t]

    def window(self, level: int) -> int:
        """Get the size of the currently open window of trailing states for the given hierarchy level."""
        return min([self.horizons[level], len(self)])

    def add(self, transition: GoalTransition) -> None:
        """ Add a state-action transition to the trace, the transition is hidden if state == next_state. """
        self.raw.append(transition)
        if not transition.degenerate:
            self.transitions.append(transition)  # Only stores a reference (no memory impact)

    def reset(self) -> None:
        """ Clear all trace variables. """
        self.raw.clear()
        self.transitions.clear()


@dataclass
class HierarchicalEligibilityTrace:
    horizon: int
    dimensions: typing.Tuple
    trace: typing.List[np.ndarray] = None  # Time trace of dim (horizon, states, goals, actions)

    def reset(self) -> None:
        if self.trace is None:
            self.trace = [np.zeros(self.dimensions, dtype=np.float32) for _ in range(self.horizon)]
        for e in self.trace:
            e[...] = 0


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
