from dataclasses import dataclass, field
from abc import ABC
import typing

import numpy as np

from utils import DotDict
from utils.game_utils import GameState
from utils.selfplay_utils import IGameHistory


@dataclass
class GoalState:
    goal: np.ndarray   # Current goal value.
    age: int           # Number of time-steps this goal has been tracked through the environment.
    achieved: bool     # Whether this goal has been achieved at the current time-step.
    atomic_index: int  # True time-measure within the current environment episode.
    subgoal_testing: bool = False

    @staticmethod
    def empty(array: np.ndarray):
        return GoalState(array, 0, False, 0)


@dataclass
class ModelFreeStats:
    rewards: float                # r^i_t: Reward given at Hierarchy i at time t.


@dataclass
class MCTSStats:
    probabilities: np.ndarray     # pi_t: Probability vector of MCTS for the action
    rewards: float                # r^i_t: Reward given at Hierarchy i at time t.
    search_returns: float         # v_t: MCTS value estimation
    observed_returns: np.ndarray  # z_t: Training targets for the value function

    opened_actions: np.ndarray    # Continuous Agents: Open actions Progressive Widening.


@dataclass
class HierarchicalGameHistory(IGameHistory):
    observations: list = field(default_factory=list)            # o_t: State Observations
    next_observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)                 # a_t+1: Action leading to transition s_t -> s_t+1
    atomic_rewards: list = field(default_factory=list)          # u_t+1: Observed env. reward after performing a_t+1
    high_level_goals: list = field(default_factory=list)
    goals: list = field(default_factory=list)                   # g_t: Goal set by hierarchy at s_t.
    goal_inference_stats: list = field(default_factory=list)    # List of either ModelFreeStats or MCTSStats
    action_inference_stats: list = field(default_factory=list)  # List of either ModelFreeStats or MCTSStats

    summed_rewards: list = field(default_factory=list)
    goal_indices: list = field(default_factory=list)
    muzero_returns: list = field(default_factory=list)

    terminated: bool = False                                    # Whether the environment has terminated

    sum_rewards: bool = False  # False = sum rewards, True = shortest path rewards.

    def __len__(self):
        return len(self.observations)

    def refresh(self) -> None:
        all([x.clear() for x in vars(self).values() if type(x) == list])
        self.terminated = False

    def capture(self, state: GameState, next_state: GameState, r: float, goal: GoalState, goal_statistics, action_statistics) -> None:
        self.observations.append(state.observation)
        self.next_observations.append(next_state.observation)
        self.atomic_rewards.append(r)

        self.actions.append(state.action)
        self.action_inference_stats.append(action_statistics)

        self.goals.append(goal)

        if goal_statistics is not None:
            self.high_level_goals.append(goal)
            self.goal_inference_stats.append(goal_statistics)
            self.goal_indices.append(goal.atomic_index)

            # Reward for higher level policy.
            if not self.sum_rewards:
                self.summed_rewards.append(-1)
            elif len(self.goal_indices) > 1:  # Sum up rewards of the terminated/ previous goal.
                # TODO: Check proper array length?
                self.summed_rewards.append(np.sum(self.atomic_rewards[self.goal_indices[-2]:self.goal_indices[-1]]))

    def terminate(self) -> None:
        self.atomic_rewards.append(0)
        self.terminated = True

        # Terminate reward for higher level policy.
        if self.sum_rewards:
            # TODO: Check proper array length?
            self.summed_rewards.append(np.sum(self.atomic_rewards[self.goal_indices[-1]:]))  # Last goal reward
        self.summed_rewards.append(0)  # Terminated reward

        # truncate goal inference and action inference
        if len(self.goal_inference_stats[-1]):
            last = self.goal_inference_stats[-1]
            self.goal_inference_stats.append((np.random.random(size=last[0].shape), np.zeros_like(last[1]), 0))
        else:
            self.goal_inference_stats.append(())

        if len(self.action_inference_stats[-1]):
            last = self.action_inference_stats[-1]
            self.action_inference_stats.append((np.random.random(size=last[0].shape), np.zeros_like(last[1]), 0))
        else:
            self.action_inference_stats.append(())

    def compute_returns(self, gamma: float = 1, n: typing.Optional[int] = None) -> None:
        """Computes the n-step returns assuming that the last recorded snapshot was a terminal state"""
        if not len(self.goal_inference_stats[-1]):
            return  # No muzero agent

        _, _, search_returns = list(zip(*self.goal_inference_stats))

        # General MDPs. Symbols follow notation from the paper.
        # z_t = u_t+1 + gamma * u_t+2 + ... + gamma^(k-1) * u_t+horizon + gamma ^ k * v_t+horizon
        # for all (t - 1) = 1... len(self) - 1
        for t in range(len(self.summed_rewards)):
            horizon = np.min([t + n, len(self.summed_rewards) - 1])

            discounted_rewards = [np.power(gamma, k - t) * self.summed_rewards[k] for k in range(t, horizon)]
            bootstrap = (np.power(gamma, horizon - t) * search_returns[horizon]) if horizon <= t + n else 0

            self.muzero_returns.append(sum(discounted_rewards) + bootstrap)

    def stackObservations(self, length: int, current_observation: typing.Optional[np.ndarray] = None,
                          t: typing.Optional[int] = None) -> np.ndarray:  # TODO: rework function.
        """ Stack the most recent 'length' elements from the observation list along the end of the observation axis """
        if length <= 1:
            if current_observation is not None:
                return current_observation
            elif t is not None:
                return self.observations[np.min([t, len(self) - 1])]
            else:
                return self.observations[-1]


def get_goal_space(net_args: DotDict, game):
    if net_args.goal_space.latent:
        return net_args.latent_depth
    else:
        return game.getDimensions()


# Hierarchical Coach Extended Functions.

def get_muzero_goal_samples(reference, histories: typing.List[HierarchicalGameHistory],
                            sample_coordinates: typing.List[typing.Tuple[int, int]],
                            sample_weight: typing.List[float]) -> typing.List:
    k = reference.args.K

    def sample(h_i, i, w):
        # TODO: Adjust i coordinate for original goal-sampled timepoint.
        i = histories[h_i].goal_indices.index(histories[h_i].goals[i].atomic_index)

        # Hindsight action transitions for goal relabelling
        indices = histories[h_i].goal_indices[i:i+k]
        actions = [histories[h_i].next_observations[j].ravel() for j in indices]
        a_truncation = k - len(actions)
        if a_truncation > 0:  # Uniform policy when unrolling beyond terminal states.
            actions += [np.random.uniform(size=actions[-1].shape) for _ in range(a_truncation)]

        inference = histories[h_i].goal_inference_stats[i:i+k+1]
        action_support, pis, _ = list(zip(*inference))

        action_support, pis = list(action_support), list(pis)

        rewards = histories[h_i].summed_rewards[i:i+k+1]
        vs = histories[h_i].muzero_returns[i:i+k+1]

        t_truncation = (k + 1) - len(pis)  # Target truncation due to terminal state
        if t_truncation > 0:
            action_support += [np.random.uniform(size=action_support[-1].shape) for _ in range(t_truncation)]
            pis += [np.zeros_like(pis[-1])] * t_truncation  # Zero vector
            rewards += [0] * t_truncation  # = 0
            vs += [0] * t_truncation  # = 0

        targets = (np.asarray(vs), np.asarray(rewards), np.asarray(pis))

        # TODO: include observations for contrastive loss.
        forward_observations = []

        return (
            histories[h_i].observations[i],  # state
            actions,                         # actions = relabelled hindsight goal transitions
            targets,
            forward_observations,
            action_support,
            sample_weight[w],
        )

    examples = [sample(h_i, i, w) for w, (h_i, i) in enumerate(sample_coordinates)]

    return examples


def get_goal_samples(reference, histories: typing.List[HierarchicalGameHistory],
                     sample_coordinates: typing.List[typing.Tuple[int, int]],
                     sample_weight: typing.List[float]) -> typing.List:
    pass


def get_muzero_action_samples(reference, histories: typing.List[HierarchicalGameHistory],
                              sample_coordinates: typing.List[typing.Tuple[int, int]],
                              sample_weight: typing.List[float]) -> typing.List:
    pass


def get_action_samples(reference, histories: typing.List[HierarchicalGameHistory],
                       sample_coordinates: typing.List[typing.Tuple[int, int]],
                       sample_weight: typing.List[float]) -> typing.List:  # TODO: Create distance-based goals?
    def sample(h_i, i):
        # Sample 50/50 HER or regular transition.
        if np.random.random() > 0.5:  # Regular transition.
            goal_achieved = reference.search_engine.goal_achieved(
                histories[h_i].observations[i], histories[h_i].goals[i].goal)

            return (
                histories[h_i].observations[i],       # state
                histories[h_i].goals[i].goal,         # goal
                histories[h_i].actions[i],            # actions
                -int(not goal_achieved),              # binary reward based on goal achieved (shortest path).
                histories[h_i].next_observations[i],  # next state
                histories[h_i].goals[i].goal,         # next goal = current goal. New goals are seen as terminal states.
                int(goal_achieved),                   # done if last observation in current goal-episode.
            )
        else:  # HER goal transition using the HAC trick (Levy et al., 2019)
            goal_episode = histories[h_i].goals[i:i+reference.args.goal_horizon+1]

            # Find goal trajectory's last observation ==> HER goal.
            j = 1
            while len(goal_episode) > j and np.all(goal_episode[j].goal == goal_episode[0].goal):
                j += 1  # j is the terminal state index for the last pursued goal.

            her_goal_achieved = (j == 1)

            return (
                histories[h_i].observations[i],           # state
                histories[h_i].next_observations[i+j-1],  # goal = last observation in goal-trajectory ==> goal reached
                histories[h_i].actions[i],                # actions
                -int(not her_goal_achieved),          # binary reward based on goal achieved.
                histories[h_i].next_observations[i],  # next state
                histories[h_i].next_observations[i+j-1],  # next goal = current goal. New goal <==> terminal state.
                int(her_goal_achieved),               # done if last observation in current goal-episode.
            )

    examples = [sample(h_i, i) for h_i, i in sample_coordinates]

    return examples
