from __future__ import annotations

import typing
import heapq
import sys
import random
from dataclasses import dataclass
from abc import ABC

import numpy as np
import gym
import tqdm

from .interface import Agent
from mazelab_experimenter.utils import find, MazeObjects, rand_argmax


class RandomAgent(Agent):
    """ The most basic of agents, one that acts uniformly random. """

    def __init__(self, observation_shape: typing.Tuple[int, int], n_actions: int, **kwargs):
        super().__init__(observation_shape=observation_shape, n_actions=n_actions)

    def sample(self, state: np.ndarray, **kwargs) -> int:
        return np.random.randint(self.n_actions)

    def reset(self) -> None:
        pass

    def update(self, **kwargs) -> None:
        pass


class TabularQLearner(Agent, ABC):

    def __init__(self, observation_shape: typing.Tuple[int, int], n_actions: int,
                 q_init: float = 0.0, **kwargs) -> None:
        super().__init__(observation_shape=observation_shape, n_actions=n_actions)

        self._q_init = q_init
        self._q_table = np.full((*observation_shape, n_actions), self._q_init, dtype=np.float32)
        self._updates = 0

    def get_updates(self) -> int:
        return self._updates

    def reset(self) -> None:
        self._q_table[...] = self._q_init
        self._updates = 0


class MonteCarloQLearner(TabularQLearner):
    """ Implements an Off-Policy Monte Carlo agent that learns a Q-table. """

    def __init__(self, observation_shape: typing.Tuple[int, int], n_actions: int, **kwargs) -> None:
        super().__init__(observation_shape=observation_shape, n_actions=n_actions, q_init=0.0)
        raise NotImplementedError("Method not yet implemented.")


class TabularQLearning(TabularQLearner):

    def __init__(self, observation_shape: typing.Tuple[int, int], n_actions: int, lr: float = 0.5,
                 epsilon: float = 0.1, discount: float = 0.95, sarsa: bool = False,
                 q_init: float = 0.0, **kwargs) -> None:
        super().__init__(observation_shape=observation_shape, n_actions=n_actions, q_init=q_init)
        self.lr = lr
        self.epsilon = epsilon
        self.discount = discount
        self._sarsa = sarsa

    def _get_pos(self, state: np.ndarray) -> np.ndarray:
        return find(a=state, predicate=lambda x: x == MazeObjects.AGENT.value)

    def sample(self, state, behaviour_policy: bool = True) -> int:
        # If acting according to exploring policy (behaviour_policy) select random action with probability epsilon
        if behaviour_policy and self.epsilon > np.random.rand():
            return np.random.randint(self.n_actions)

        # Otherwise act greedy with respect to the current Q-table
        pos = self._get_pos(state)  # 1D 2-element array 
        argmaxes = np.where(self._q_table[pos] == np.max(self._q_table[pos]))[0]  # 1D array of max length self.n_actions

        return np.random.choice(argmaxes)

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, **kwargs) -> typing.Optional[typing.Any]:
        # Get bootstrap-action either with the behaviour policy (SARSA) or with the optimal policy (Q-Learning) 
        next_action = self.sample(next_state, behaviour_policy=self._sarsa)

        # Get indices within Q-table corresponding to the given states.
        pos_t, pos_next = self._get_pos(state), self._get_pos(next_state)  # 1D 2-element arrays

        # Q-learning update
        bootstrap = (1 - int(done)) * self._q_table[pos_next][next_action]
        self._q_table[pos_t][action] += self.lr * (reward + self.discount * bootstrap - self._q_table[pos_t][action])
        self._updates += 1

    def train(self, _env: gym.Env, num_episodes: int, progress_bar: bool = False) -> None:
        iterator = (tqdm.trange(num_episodes, file=sys.stdout, desc="TabularQLearning Training")
                    if progress_bar else range(num_episodes))

        for _ in iterator:
            # Reinitialize environment after each episode.
            state, goal_achieved, done = _env.reset(), False, False

            while not done:
                # Update to the next state.
                a = self.sample(state)
                next_state, r, done, meta = _env.step(a)

                # Annotate an episode as done if the agent is actually in a goal-state (not if the time expires).
                goal_achieved = meta['goal_achieved']

                # Perform Q-learning update and update state of control
                self.update(state, a, r, next_state, goal_achieved, meta=meta)
                state = next_state

            # Cleanup environment variables
            _env.close()


class TabularQLearningN(TabularQLearning):
    """ Simple n-step SARSA and n-step Tree Backup implementation of Sutton and Barto 2018. """

    @dataclass
    class Transition:
        state: np.ndarray
        action: int
        reward: float
        next_state: np.ndarray
        terminal: bool

    def __init__(self, observation_shape: typing.Tuple[int, int], n_actions: int, lr: float = 0.5, n_steps: int = 1,
                 epsilon: float = 0.1, discount: float = 0.95, sarsa: bool = False, optimal: bool = True,
                 q_init: float = 0.0, **kwargs) -> None:
        super().__init__(observation_shape=observation_shape, n_actions=n_actions, lr=lr,
                         epsilon=epsilon, discount=discount, sarsa=sarsa, q_init=q_init)

        self.optimal = optimal  # Whether to use Q* for TB-updates or a full Q expectation

        self.n_steps = n_steps  # Uses Tree-Backup(n) for sarsa = False, otherwise simply n-step SARSA.
        self.transitions = list()

    def tree_backup(self, transitions: typing.List[TabularQLearningN.Transition], t_start: int) -> None:
        # Basic n-step Tree Backup
        transition_t = transitions[t_start]
        transition_T = transitions[-1]

        # Get indices within Q-table corresponding to the given states.
        pos_start = self._get_pos(transitions[t_start].state)  # 1D 2-element arrays

        if transition_T.terminal:
            G = transition_T.reward
        else:
            pos_end = self._get_pos(transition_T.next_state)

            q_end = self._q_table[pos_end].max()
            if not self.optimal:  # Reweigh Q using soft epsilon policy.
                q_end = self.epsilon * (self._q_table[pos_end].mean()) + (1 - self.epsilon) * q_end

            G = transition_T.reward + self.discount * q_end

        # Backup from leaf node, cut traces where a != a* if using Q-learning.
        for k in reversed(range(t_start, len(transitions) - 1)):
            # Awkward indexing due to r_t being stored under transition_t.
            s_k, a_k = transitions[k + 1].state, transitions[k + 1].action

            if not self._sarsa:  # Use Tree-Backup Algorithm.
                pos_k = self._get_pos(s_k)  # 1D 2-element arrays

                # Reset trace if selected action isn't Q-optimal (don't reset in case of ties).
                q_max = self._q_table[pos_k].max()
                greedy = np.isclose(q_max, self._q_table[pos_k][a_k])
                if self.optimal:
                    G = G if greedy else q_max
                else:  # Reweigh Q using soft epsilon policy.
                    mean = np.delete(self._q_table[pos_k], a_k).sum() / self.n_actions
                    prob = (1 - self.epsilon) * int(greedy) + self.epsilon / self.n_actions

                    G = self.epsilon * mean + prob * G
                    if not greedy:  # Add remainder (1 - epsilon) mass with Q-max
                        G += (1 - self.epsilon) * q_max

            # Generalizes n-step SARSA.
            G = transitions[k].reward + self.discount * G

        # Q-learning update
        self._q_table[pos_start][transition_t.action] += self.lr * (G - self._q_table[pos_start][transition_t.action])
        self._updates += 1

    def update(self, transitions: typing.List[TabularQLearningN.Transition], t: int, **kwargs) -> None:
        tau = t - self.n_steps + 1
        if tau >= 0:
            for i in range(tau, tau + self.n_steps ** int(transitions[-1].terminal)):
                self.tree_backup(transitions, i)

    def train(self, _env: gym.Env, num_episodes: int, progress_bar: bool = False) -> None:
        iterator = (tqdm.trange(num_episodes, file=sys.stdout, desc="TabularQLearning Training")
                    if progress_bar else range(num_episodes))

        for _ in iterator:
            # Reinitialize environment after each episode.
            state, goal_achieved, done = _env.reset(), False, False

            # n-step buffer
            self.transitions.clear()

            t = 0
            while not done:
                # Update to the next state.
                a = self.sample(state)
                next_state, r, done, meta = _env.step(a)

                # Annotate an episode as done if the agent is actually in a goal-state (not if the time expires).
                goal_achieved = meta['goal_achieved']

                # Perform n-step Q-learning update.
                self.transitions.append(TabularQLearningN.Transition(state, a, r, next_state, goal_achieved))
                self.update(self.transitions, t, meta=meta)

                # Update state of control
                state = next_state
                t += 1

            # Cleanup environment variables
            _env.close()


class TabularQLambda(TabularQLearningN):
    """ Implements a simple Tree-Backup(λ) agent (subsuming SARSA(λ)) optionally with replacing traces. """

    def __init__(self, observation_shape: typing.Tuple[int, int], n_actions: int, lr: float = 0.5, decay: float = 0.9,
                 epsilon: float = 0.1, discount: float = 0.95, replace: bool = False, decay_lr: float = 0.0,
                 sarsa: bool = False, optimal: bool = True, q_init: float = 0.0, **kwargs) -> None:
        super().__init__(observation_shape=observation_shape, n_actions=n_actions, lr=lr, n_steps=None,
                         epsilon=epsilon, discount=discount, sarsa=sarsa, optimal=optimal, q_init=q_init)
        # Learning rate scheduling (Q-values may diverge if not used)
        self.decay_lr = decay_lr
        self._lr_base = lr

        self.decay = decay
        self.replace = replace  # Whether to use replacing traces or accumulating traces.
        self.trace = np.zeros_like(self._q_table, dtype=np.float32)

        self.episodes = 0

    def update_step_sizes(self) -> None:
        self.lr = self._lr_base * (self.episodes ** self.decay_lr)

    def reset(self) -> None:
        super().reset()
        self.lr = self._lr_base
        self.episodes = 0

    def update(self, transition: TabularQLearningN.Transition, **kwargs) -> typing.Optional[typing.Any]:
        # Get indices within Q-table and trace corresponding to the given states.
        pos_t, pos_next = self._get_pos(transition.state), self._get_pos(transition.next_state)  # 1D 2-element arrays
        # Get bootstrap and update actions.
        a_t, a_t_next = transition.action, self.sample(transition.next_state, behaviour_policy=self._sarsa)

        # Update eligibility trace. SARSA(λ) for retain = 1, TB(λ) for retain = 0.
        retain = 1
        if not self._sarsa:  # TB(λ)
            q_max = self._q_table[pos_t].max()
            greedy = (q_max == self._q_table[pos_t][a_t])

            # If Optimal: reset trace if action not Q-optimal (ignore ties).
            retain = int(greedy)
            if not self.optimal:  # Else: Expected sarsa --> retain = Pr(a | s)
                retain = (1 - self.epsilon) * int(greedy) + self.epsilon / self.n_actions

        # Replacing traces: Compute z_t = γλπ(a|s)z_(t-1) + ∇q(s, a, w), w are the 'weights' == Q-table values.
        self.trace = self.discount * self.decay * retain * self.trace
        if self.replace:  # Reset trace on revisits.
            # self.trace[pos_t][...] = 0  # OPTIONAL: Clear trace of other actions too.
            self.trace[pos_t][a_t] = 1
        else:  # Accumulate trace
            self.trace[pos_t][a_t] += 1

        # Compute the Bellman error
        delta = transition.reward + self.discount * self._q_table[pos_next][a_t_next] - self._q_table[pos_t][a_t]

        # SARSA(λ) update to critic parameters w.
        self._q_table += self.lr * delta * self.trace

    def train(self, _env: gym.Env, num_episodes: int, progress_bar: bool = False) -> None:
        iterator = (tqdm.trange(num_episodes, file=sys.stdout, desc="TabularQLearning Training")
                    if progress_bar else range(num_episodes))

        for _ in iterator:
            # Reinitialize environment after each episode.
            state, goal_achieved, done = _env.reset(), False, False
            self.episodes += 1

            # Initialize agent dependencies
            self.trace[...] = 0
            self.transitions.clear()
            self.update_step_sizes()

            t = 0
            while not done:
                # Update to the next state.
                a = self.sample(state)
                next_state, r, done, meta = _env.step(a)

                # Annotate an episode as done if the agent is actually in a goal-state (not if the time expires).
                goal_achieved = meta['goal_achieved']

                # Perform n-step Q-learning update.
                self.transitions.append(TabularQLearningN.Transition(state, a, r, next_state, goal_achieved))
                self.update(self.transitions[-1], meta=meta)

                # Update state of control
                state = next_state
                t += 1

            # Cleanup environment variables
            _env.close()


class TabularQET(TabularQLambda):

    def __init__(self, observation_shape: typing.Tuple[int, int], n_actions: int, lr: float = 0.5, decay: float = 0.9,
                 epsilon: float = 0.1, discount: float = 0.95, replace: bool = False, optimal: bool = True,
                 decay_lr: float = -0.5, eta: float = 0.0, beta: float = 0.5, sarsa: bool = False,
                 q_init: float = 0.0, **kwargs) -> None:
        super().__init__(observation_shape=observation_shape, n_actions=n_actions, lr=lr, n_steps=None, decay=decay,
                         decay_lr=decay_lr, replace=replace, epsilon=epsilon, discount=discount, sarsa=sarsa,
                         optimal=optimal, q_init=q_init)
        # Learning rate scheduling for trace model
        self._beta_base = beta

        self.eta = eta    # Interpolation variable between eligibility trace and expected trace.
        self.beta = beta  # Expected trace learning rate.
        self.trace_model = np.zeros((*self.trace.shape, *self.trace.shape), dtype=np.float32)

    def update_step_sizes(self) -> None:
        self.lr = self._lr_base * (self.episodes ** self.decay_lr)
        self.beta = self._beta_base * (self.episodes ** self.decay_lr)

    def reset(self) -> None:
        super().reset()
        self.lr = self._lr_base
        self.beta = self._beta_base
        self.trace_model[...] = 0

    def update(self, transition: TabularQLearningN.Transition, **kwargs) -> typing.Optional[typing.Any]:
        # Get indices within Q-table and trace corresponding to the given states.
        pos_t, pos_next = self._get_pos(transition.state), self._get_pos(transition.next_state)  # 1D 2-element arrays
        # Get bootstrap and update actions.
        a_t, a_t_next = transition.action, self.sample(transition.next_state, behaviour_policy=self._sarsa)

        # Update eligibility trace.
        retain = 1
        if not self._sarsa:  # TB(λ)
            q_max = self._q_table[pos_t].max()
            greedy = np.isclose(q_max, self._q_table[pos_t][a_t])

            # If Optimal: reset trace if action not Q-optimal (ignore ties).
            retain = int(greedy)
            if not self.optimal:  # Else: Expected sarsa --> retain = Pr(a | s)
                retain = (1 - self.epsilon) * int(greedy) + self.epsilon / self.n_actions

        # Replacing traces: Compute z_t = γλπ(a|s)z_(t-1) + ∇q(s, a, w), w are the 'weights' == Q-table averages.
        self.trace = self.discount * self.decay * retain * self.trace
        if self.replace:  # Reset trace on revisits.
            # self.trace[pos_t][...] = 0  # OPTIONAL: Clear trace of other actions too.
            self.trace[pos_t][a_t] = 1
        else:  # Accumulate trace
            self.trace[pos_t][a_t] += 1

        # Compute the TD error
        delta = transition.reward + self.discount * self._q_table[pos_next][a_t_next] - self._q_table[pos_t][a_t]

        # Update the Expected Eligibility trace Model.
        self.trace_model[pos_t][a_t] += self.beta * (self.trace - self.trace_model[pos_t][a_t])

        # ET(λ, η) update for the critic parameters w.
        ys = (1 - self.eta) * self.trace_model[pos_t][a_t] + self.eta * self.trace
        self._q_table += self.lr * delta * ys


class TabularDynaQ(TabularQLearning):
    """
    A deterministic tabular (Prioritized Sweeping) DynaQ agent as described by Sutton et al., 2018
    Chapter 8.2, 8.3, and 8.4.
    """

    def __init__(self, observation_shape: typing.Tuple[int, int], n_actions: int, n_iter: int = 10,
                 priority: float = 0.0, lr: float = 0.5, epsilon: float = 0.1, discount: float = 0.95,
                 q_init: float = 0.0, **kwargs) -> None:
        super().__init__(observation_shape=observation_shape, n_actions=n_actions, lr=lr,
                         epsilon=epsilon, discount=discount, sarsa=False, q_init=q_init)

        # Initialize model and sweeping parameters.
        self._model_iter = n_iter
        self._priority = priority
        self._model = dict()

        # If specified, initialize Prioritized Sweeping parameters.
        if self._priority:
            self._backward_model = dict()
            self._queue = []  # Maintained as a priority queue with heapq.

    def reset(self) -> None:
        super().reset()
        self._model.clear()
        if self._priority:
            self._backward_model.clear()
            self._queue.clear()

    def _store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        # Get hashable state and next state coordinates.
        pos_t, pos_next = tuple(self._get_pos(state)), tuple(self._get_pos(next_state))

        # Store transition in model.
        if pos_t not in self._model:
            self._model[pos_t] = dict()
        self._model[pos_t][action] = (reward, pos_next, done)

        # Store backwards-transition in the prioritized-sweeping model.
        if self._priority:
            if pos_next not in self._backward_model:
                self._backward_model[pos_next] = dict()
            self._backward_model[pos_next][(pos_t, action)] = reward

    def _dyna_update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        
        """
        # >> super().update: DynaQ first updates Q-table using canonical one-step Q-learning transition (Chapter 8.2).
        next_action = self.sample(next_state, behaviour_policy=self._sarsa)
        pos_t, pos_next = self._get_pos(state), self._get_pos(next_state)  # 1D 2-element arrays
        bootstrap = (1 - int(done)) * self._q_table[pos_next][next_action]
        self._q_table[pos_t][action] += self.lr * (reward + self.discount * bootstrap - self._q_table[pos_t][action])

        for _ in range(self._model_iter):
            s = random.choice(list(self._model))
            a = random.choice(list(self._model[s]))

            # Retrieve experience from model.
            r, s_next, terminal = self._model[s][a]
            a_next = np.random.choice(np.where(self._q_table[s_next] == np.max(self._q_table[s_next]))[0])

            # Update Q-table based on retrieved experience.
            self._q_table[s][a] += self.lr * (r + self.discount * (1 - int(terminal)) * self._q_table[s_next][a_next] - self._q_table[s][a])

    def _prioritized_sweeping(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        next_action = self.sample(next_state, behaviour_policy=self._sarsa)
        pos_t, pos_next = self._get_pos(state), self._get_pos(next_state)  # 1D 2-element arrays

        # Insert transition into priority queue if td_error exceeds magnitude threshold.
        td_error = reward + (1 - int(done)) * self._q_table[pos_next][next_action] - self._q_table[pos_t][action]
        if abs(td_error) > self._priority:
            # Priotity is the absolute td-error, the priority is negated for the queue ranking.
            heapq.heappush(self._queue, (-abs(td_error), (pos_t, action)))

        i = 0
        while i < self._model_iter and len(self._queue) > 0:
            i += 1

            # Pop state-action pair with highest priority (highest TD-error)
            p, (s, a) = heapq.heappop(self._queue)

            # Retrieve experience from model.
            r, s_next, terminal = self._model[s][a]
            a_next = np.random.choice(np.where(self._q_table[s_next] == np.max(self._q_table[s_next]))[0])

            # Update Q-table based on retrieved experience.
            self._q_table[s][a] += self.lr * (r + self.discount * (1 - int(terminal)) * self._q_table[s_next][a_next] - self._q_table[s][a])

            if s in self._backward_model:
                # Sweep through the backwards model and update the priority queue accordingly.
                for (s_prev, a_prev), r_t in self._backward_model[s].items():
                    # Compute the greedy backwards TD-error
                    a_backwards = np.random.choice(np.where(self._q_table[s] == np.max(self._q_table[s]))[0])
                    backward_td_error = r_t + self.discount * self._q_table[s][a_backwards] - self._q_table[s_prev][a_prev]

                    # Insert back into priority queue if td_error exceeds magnitude threshold.
                    if abs(backward_td_error) > self._priority:
                        heapq.heappush(self._queue, (-abs(backward_td_error), (s_prev, a_prev)))

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, **kwargs) -> None:
        """ """
        # TODO: correct state and next_state for Environment removal due to Agent.VALUE overriden by Goal.VALUE.
        #  use 'meta' kwarg to correct this using the environment meta-information.

        self._store_transition(state=state, action=action, reward=reward, next_state=next_state, done=done)

        if self._priority:  # Uses Prioritized Sweeping if self._priority float value is not 0.0
            self._prioritized_sweeping(state=state, action=action, reward=reward, next_state=next_state, done=done)
        else:  # Otherwise use DynaQ.
            self._dyna_update(state=state, action=action, reward=reward, next_state=next_state, done=done)

        self._updates += 1
