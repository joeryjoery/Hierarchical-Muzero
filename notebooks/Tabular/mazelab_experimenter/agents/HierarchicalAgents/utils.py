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

    def window(self, level: int, raw: bool = False) -> int:
        """Get the size of the currently open window of trailing states for the given hierarchy level."""
        return min([self.horizons[level], (len(self) if (not raw) else len(self.raw))])

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

    @dataclass
    class TraceEntry:
        state: int
        action: int
        time: int
        previous: HierarchicalEligibilityTrace.TraceEntry = None

    horizon: int
    num_goals: int
    trace: typing.List[typing.List] = None
    cuts: typing.List[np.ndarray] = None   # Time-index for each goal to track where traces are cut.
    TRUNCATE: float = 1e-8  # Trace truncation value for extremely small floats (as good as zero)

    def add(self, h: int, s: int, a: int, t: int) -> None:
        entry = self.TraceEntry(state=s, action=a, time=t)
        if len(self.trace[h]):
            entry.previous = self.trace[h][-1]
        self.trace[h].append(entry)

    def cut(self, h: int, indices: np.ndarray, t: int) -> None:
        self.cuts[h][indices] = t

    def reset(self) -> None:
        """ Clear/ initialize all trace variables. """
        if self.trace is None:
            self.trace = [list() for _ in range(self.horizon)]
            self.cuts = [np.zeros(self.num_goals, dtype=np.int32) for _ in range(self.horizon)]

        for i in range(self.horizon):
            self.trace[i].clear()
            self.cuts[i][...] = 0


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
