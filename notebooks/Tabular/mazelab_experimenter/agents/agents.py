import typing
import heapq
import sys
import random
from abc import ABC

import numpy as np
import gym
import tqdm

from .interface import Agent
from mazelab_experimenter.utils import find, MazeObjects


class RandomAgent(Agent):
    """ The most basic of agents, one that acts uniformly random. """

    def __init__(self, observation_shape: typing.Tuple[int, int], n_actions: int):
        super().__init__(observation_shape=observation_shape, n_actions=n_actions)

    def sample(self, state: np.ndarray, **kwargs) -> int:
        return np.random.randint(self.n_actions)

    def reset(self) -> None:
        pass

    def update(self, **kwargs) -> None:
        pass


class TabularQLearner(Agent, ABC):

    def __init__(self, observation_shape: typing.Tuple[int, int], n_actions: int, q_init: float = 0.0) -> None:
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

    def __init__(self, observation_shape: typing.Tuple[int, int], n_actions: int):
        super().__init__(observation_shape=observation_shape, n_actions=n_actions, q_init=q_init)


class TabularQLearning(TabularQLearner):

    def __init__(self, observation_shape: typing.Tuple[int, int], n_actions: int, lr: float = 0.5,
                 epsilon: float = 0.1, discount: float = 0.95, sarsa: bool = False, q_init: float = 0.0) -> None:
        super().__init__(observation_shape=observation_shape, n_actions=n_actions, q_init=q_init)
        self.lr = lr
        self.epsilon = epsilon
        self.discount = discount
        self._sarsa = sarsa

    def _get_pos(self, state: np.ndarray) -> np.ndarray:
        return find(a=state, predicate=lambda x: x == MazeObjects.AGENT.value)

    def sample(self, state, behaviour_policy: bool = True) -> int:
        # If acting according to exploring policy (behaviour_policy) select random action with probability epsilon
        if behaviour_policy and self.epsilon < np.random.rand():
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


class TabularDynaQ(TabularQLearning):
    """ A deterministic tabular (Prioritized Sweeping) DynaQ agent as described by Sutton et al., 2018 Chapter 8.2, 8.3, and 8.4. """

    def __init__(self, observation_shape: typing.Tuple[int, int], n_actions: int, n_iter: int = 10,
                 priority: float = 0.0, lr: float = 0.5, epsilon: float = 0.1, discount: float = 0.95,
                 q_init: float = 0.0) -> None:
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
