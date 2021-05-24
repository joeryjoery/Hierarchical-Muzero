from __future__ import annotations
import typing
from dataclasses import dataclass, field

import numpy as np
import gym

from ..HierQV2 import HierQV2
from mazelab_experimenter.utils import rand_argmax


class HierQTS(HierQV2):

    @dataclass
    class Node:
        state: int
        goal: int
        visits: int
        value: float
        parent: typing.Optional[HierQTS.Node] = None
        children: typing.List[int] = field(default_factory=list)
        expanded: typing.Dict[int, HierQTS.Node] = field(default_factory=dict)

    @dataclass
    class MinMaxBounds:
        min: float
        max: float

        def update(self, value: float) -> float:
            self.min = min([self.min, value])
            self.max = max([self.max, value])

    def __init__(self, observation_shape: typing.Tuple, n_actions: int, n_levels: int,
                 horizons: typing.Union[typing.List[int]], lr: float = 0.5, epsilon: float = 0.1,
                 discount: float = 0.95, relative_actions: bool = True, universal_top: bool = True,
                 ignore_training_time: bool = False, legal_states: np.ndarray = None,
                 planning_level: int = 1, num_sims: int = 10, c_uct: float = 1.41, temp: float = 1.0,
                 simulator: gym.Env = None) -> None:
        super().__init__(observation_shape=observation_shape, n_actions=n_actions, n_levels=n_levels,
                         horizons=horizons, lr=lr, epsilon=epsilon, discount=discount,
                         relative_actions=relative_actions, universal_top=universal_top,
                         ignore_training_time=ignore_training_time, legal_states=legal_states)

        # Tree Search Variables.
        self.planning_level = planning_level
        self.num_sims = num_sims
        self.c_uct = c_uct
        self.temp = temp
        self.simulator = simulator  # Should be a copy of the gym.Env object used for the agent.
        self.model = dict()

    def _select(self, node: HierQTS.Node, level: int, bounds: HierQTS.MinMaxBounds) -> int:
        """Select an action based on UCT for tree traversal. """
        vs, ns = np.zeros_like(node.children), np.zeros_like(node.children)
        gc = node.goal * int(self.critics[level].goal_conditioned)
        for i, c in enumerate(node.children):
            if c not in node.expanded:  # Initialize children with N(s, a) = 1, Q_tree(s, a) = Q_table(s,a)
                ns[i] = 1
                vs[i] = self.critics[level].table[node.state, gc, i]
            else:
                ns[i] = node.expanded[c].visits
                vs[i] = node.expanded[c].value / node.expanded[c].visits

        # Minmax normalize Exploitation term using tree bounds (MuZero trick). Clip if min = max.
        qs = np.clip((vs - bounds.min) / (bounds.max - bounds.min + 1e-4), 0, 1)

        # Out of bounds check for probability bins.
        coords = self.S_xy[node.state] + self.A_dxy[level]
        legals = np.all(0 <= coords, axis=-1) & np.all(coords < self.observation_shape, axis=-1)
        legals[len(legals) // 2] = 0  # Do nothing mask.
        p = legals / legals.sum()  # TODO Contextual weighting?

        # Compute PUCT formula
        uct = qs + self.c_uct * p * np.sqrt(np.log(node.visits) / ns)

        # Sample an action given the UCT bounds with preference for the node's goal when breaking ties.
        pref = None
        if HierQV2.inside_radius(self.S_xy[node.state], self.G_xy[level][node.goal],
                                 r=self.G_radii[level - 1], motion=self.motion):
            pref = self.ravel_delta_indices(
                center_deltas=np.diff([self.S_xy[node.state], self.G_xy[level][node.goal]]).T,
                r=self.G_radii[level - 1], motion=self.motion).item()

        action = rand_argmax(np.asarray(uct), preference=pref, mask=legals)

        return action

    def _step(self, node: HierQTS.Node, action: int, level: int) -> HierQTS.Node:
        # Selection --> Tree traversal
        if action in node.expanded:
            return node.expanded[action]
        # Else: Selection --> Expansion.

        goal = self._get_index(self.S_xy[node.state] + self.A_dxy[level][action])
        if self.simulator is None:
            # Take a temporally extended step inside the Q-table.
            new_state = goal
        elif self.n_levels == 2:
            # Initialize simulator to execute the temporally extended action (subgoal) in the actual environment.
            self.simulator.reset()
            self.simulator.maze.objects.agent.positions[0] = self.S_xy[node.state]

            # Simulate subgoal action following the greedy atomic policy
            s = node.state
            for _ in range(self.horizons[0]):
                a = self.get_level_action(s, goal, level - 1, explore=False)
                *_, meta = self.simulator.step(a)
                s = self._get_index(meta['coord_next'])

                if s == goal or s == node.goal:
                    break

            # Cleanup
            self.simulator.close()
            new_state = s
        else:
            raise NotImplementedError("")

        # Expansion: Empty Node (uninitialized)
        empty = HierQTS.Node(new_state, node.goal, visits=0, value=0, parent=node, children=self.A[level])
        node.expanded[action] = empty

        return node.expanded[action]

    def _save(self, state: int, goal: int, level: int) -> HierQTS.Node:
        """Perform Tree Search for improving Q estimations.

        SAVE refers to Search with Amortized Value Estimates: https://arxiv.org/abs/1912.02807
        The (simple) idea that we transfer from this paper is the Q_tree(s, a) initialization of
        search tree edges with their critic Q_table(s, a) values and the V_leaf(s) = max_a Q(s, a) bootstrap.

        """
        # Initialize Search Tree
        gc = goal * int(self.critics[level].goal_conditioned)
        node = root = HierQTS.Node(state, goal, visits=1, value=self.critics[level].table[state, gc].max(),
                                   parent=None, children=self.A[level])
        bounds = HierQTS.MinMaxBounds(self.critics[level].table[state, gc].min(), 0)

        # Perform Tree Search guided by UCT with a Greedy Q-Table Backup.
        for n in range(self.num_sims - 1):
            # Selection/ Expansion until leaf node.
            while node.visits > 0 and node.state != root.goal:
                a = self._select(node, level, bounds)  # Q-function informed PUCT action selection
                node = self._step(node, a, level)  # Step forward in the simulation/ expand tree

            # Rollout/ Bootstrap --> Greedy Q-backup
            gc = node.goal * int(level != self.n_levels - 1 or self.universal_top)
            value = (-1 + self.critics[level].table[node.state, gc].max()) * int(node.state != root.goal)

            # Back up values
            while node:
                node.visits += 1
                node.value += value
                value = -1 + self.discount * value
                # print('\n', n, node)
                # Update bounds and back up to parent node.
                bounds.update(value)
                node = node.parent

            # Reset trace.
            node = root

        # Return root's children visit-counts and root-value.
        return root

    def get_level_action(self, s: int, g: int, level: int, explore: bool = True) -> int:
        if level == 0 or self.num_sims <= 1:  # Use direct Q-table lookup for sampling actions.
            return super().get_level_action(s, g, level, False)
        # Else: use a UCT-guided Tree Search for action selection for goals.

        # Sample an action from the Tree Search results.
        root = self._save(s, g, level)  # Returns a data structure containing Tree Search results.

        gc = g * int(self.critics[level].goal_conditioned)
        qs = np.asarray([(root.expanded[c].value / root.expanded[c].visits if c in root.expanded
                          else self.critics[level].table[s, gc, c])
                         for i, c in enumerate(root.children)])

        c = self.S_xy[s] + self.A_dxy[level]
        mask = np.all(0 <= c, axis=-1) & np.all(c < self.observation_shape, axis=-1)
        mask[len(mask) // 2] = 0  # Mask do nothing

        a_i = rand_argmax(qs, mask=mask) if (self.epsilon < np.random.rand() or not explore) else rand_argmax(mask)
        action_state = self._get_index(c[a_i])

        #
        # print()
        # print('before', self.critics[level].table[s, gc])
        # print('after', qs)
        # print()
        # # #
        # counts = np.asarray([(root.expanded[c].visits if c in root.expanded else 0) for c in root.children])
        # probabilities = (counts ** (1.0 / self.temp)) / (np.sum(counts) ** (1.0 / self.temp))
        # a_i = np.argmax(counts) if self.temp == 0.0 else np.random.choice(len(probabilities), p=probabilities)

        # Return absolute state as an action.
        # action_state = root.expanded[root.children[a_i]].state
        # print(action_state)
        if not explore:
            return action_state

        return action_state, qs

    def update_critic(self, level: int, s: int, a: int, s_next: int, end_pos: int = None) -> None:
        super().update_critic(level, s, a, s_next, end_pos)

        # Model update from MCTS
        if level and s in self.model:
            print(self.model.keys())
            self.critics[level].table[s, 0] += self.lr / self.model[s][1] * (self.model[s][0] - self.critics[level].table[s, 0])
            self.model[s][1] += 1

