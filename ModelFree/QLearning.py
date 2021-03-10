import typing

import numpy as np
import tensorflow as tf

from ModelFree.Interface import IUVFA
from utils.loss_utils import scalar_loss, cast_to_tensor, safe_l2norm
from utils.HierarchicalUtils import get_goal_space, GoalState


class UDQN(IUVFA):
    """ Double Q-learning inspired DQN agent. """

    def __init__(self, game, net_args, architecture: str, goal_sampling: bool = False):
        super().__init__(game, net_args, f'DQN_{architecture}', goal_sampling)

        if goal_sampling:
            raise NotImplementedError("Discrete goal sampling specified through DQN architecture...")

        self.loss_function = tf.keras.losses.Huber()
        assert hasattr(self.network, 'model') and hasattr(self.network, 'target'), "Incorrect network specification."

        # Select parameter optimizer from config.
        if self.net_args.optimizer.method == "adam":
            self.optimizer = tf.optimizers.Adam(lr=self.net_args.optimizer.lr_init)
        elif self.net_args.optimizer.method == "sgd":
            self.optimizer = tf.optimizers.SGD(lr=self.net_args.optimizer.lr_init,
                                               momentum=self.net_args.optimizer.momentum)
        else:
            raise NotImplementedError(f"Optimization method {self.net_args.optimizer.method} not implemented...")

        # Set default goal to zeros ==> defaults to no goal/ the canonical algorithm.
        self.set_goal(GoalState.empty(np.zeros(get_goal_space(net_args, game))))

    def sample(self, observation: np.ndarray, epsilon: float = 0.01, **kwargs) -> float:
        action_bins = self.network.model.predict([observation[np.newaxis, ...],
                                                  self.current_goal.goal[np.newaxis, ...]])[0]

        if np.random.rand() < epsilon:
            return np.random.randint(action_bins.shape[-1])
        else:
            return np.argmax(action_bins)

    def train(self, examples: typing.List):
        states, goals, actions, rewards, next_states, next_goals, done = list(zip(*examples))

        # Cast to arrays.
        states, goals, actions, rewards, next_states, next_goals, done = [np.asarray(x) for x in [
            states, goals, actions, rewards, next_states, next_goals, done]]

        action_ohe = np.zeros(shape=(len(states), self.network.action_size))
        action_ohe[np.arange(len(states)), actions] = 1

        # update
        q_next = self.network.model.predict_on_batch([next_states, next_goals])

        target_actions = np.argmax(q_next, axis=-1)

        q_target = self.network.target.predict_on_batch([next_states, next_goals])
        bootstrap = q_target[np.arange(len(target_actions)), target_actions]

        y = rewards + self.net_args.gamma * (1 - done) * bootstrap
        loss = self._update_model(cast_to_tensor(states), cast_to_tensor(goals), cast_to_tensor(action_ohe),
                                  cast_to_tensor(y))

        if self.steps % self.net_args.target_frequency == 0:
            # update target networks
            for (a, b) in zip(self.network.target.weights, self.network.model.weights):
                a.assign(b)

        self.monitor.log(loss, 'DQN_loss')
        self.steps += 1

    @tf.function
    def _update_model(self, states, goals, actions, targets):
        with tf.GradientTape() as tape:
            qs = self.network.model([states, goals])

            loss = self.loss_function(targets, tf.reduce_sum(qs * actions, axis=-1))

            l2 = tf.reduce_sum([safe_l2norm(x) for x in self.network.model.weights])
            loss += l2 * self.net_args.l2

        grads = tape.gradient(loss, self.network.model.weights)
        self.optimizer.apply_gradients(zip(grads, self.network.model.weights))

        return loss

    def save_checkpoint(self, folder: str = 'checkpoint', filename: str = 'checkpoint.pth.tar') -> None:
        pass

    def load_checkpoint(self, folder: str = 'checkpoint', filename: str = 'checkpoint.pth.tar') -> None:
        pass


class UDDPG(IUVFA):
    """ TD3-UVFA inspired DDPG agent """

    def __init__(self, game, net_args, architecture: str, goal_sampling: bool = False):
        super().__init__(game, net_args, f'DDPG_{architecture}', goal_sampling)

        assert hasattr(self.network, 'actor'), "No actor specified."
        assert hasattr(self.network, 'critic_a') and hasattr(self.network, 'critic_b'), \
            "Incorrect critic specification."
        assert hasattr(self.network, 'actor_target'), "No target actor specified."
        assert hasattr(self.network, 'critic_target_a') and hasattr(self.network, 'critic_target_b'), \
            "Incorrect target critic specification."

        # Select parameter optimizer from config.
        if self.net_args.optimizer.method == "adam":
            self.actor_optimizer = tf.optimizers.Adam(lr=self.net_args.optimizer.lr_init)
            self.critic_optimizer_a = tf.optimizers.Adam(lr=self.net_args.optimizer.lr_init)
            self.critic_optimizer_b = tf.optimizers.Adam(lr=self.net_args.optimizer.lr_init)
        elif self.net_args.optimizer.method == "sgd":
            self.actor_optimizer = tf.optimizers.SGD(lr=self.net_args.optimizer.lr_init,
                                                     momentum=self.net_args.optimizer.momentum)
            self.critic_optimizer_a = tf.optimizers.SGD(lr=self.net_args.optimizer.lr_init,
                                                        momentum=self.net_args.optimizer.momentum)
            self.critic_optimizer_b = tf.optimizers.SGD(lr=self.net_args.optimizer.lr_init,
                                                        momentum=self.net_args.optimizer.momentum)
        else:
            raise NotImplementedError(f"Optimization method {self.net_args.optimizer.method} not implemented...")

        # Set default goal to zeros ==> defaults to no goal/ the canonical algorithm.
        self.set_goal(GoalState.empty(np.zeros(self.network.get_goal_space(net_args))))

    def sample(self, observation: np.ndarray, epsilon: float = 1.0, **kwargs) -> np.ndarray:
        # Inference
        mu = self.network.actor.predict([observation[np.newaxis, ...], self.current_goal.goal[np.newaxis, ...]])[0]

        # Post-processing
        mu += np.random.normal(0, np.ones_like(mu) * epsilon)
        mu = np.clip(mu, 0, 1)

        return mu

    def train(self, examples: typing.List):
        states, goals, actions, rewards, next_states, next_goals, done = list(zip(*examples))

        # Cast to arrays.
        states, goals, actions, rewards, next_states, next_goals, done = [np.asarray(x) for x in [
            states, goals, actions, rewards, next_states, next_goals, done]]

        # Compute forward Q-learning actions using TD3 target policy smoothing.
        target_actions = self.network.actor_target.predict_on_batch([next_states, next_goals])

        noise = np.random.normal(0, np.ones_like(target_actions) * self.net_args.noise)
        target_actions = np.clip(target_actions + np.clip(noise, -self.net_args.c, self.net_args.c), 0, 1)

        # Compute TD3 double Q-learning bootstrap.
        bootstrap = np.min([
            self.network.critic_target_a.predict_on_batch([next_states, target_actions, next_goals]),
            self.network.critic_target_b.predict_on_batch([next_states, target_actions, next_goals])
        ], axis=0)

        # Define critic's learning targets.
        y = rewards + self.net_args.gamma * (1 - done) * bootstrap

        # Update both critic's parameters.
        loss_critic_a = self._update_critic(states, goals, actions, y,
                                            self.network.critic_a, self.critic_optimizer_a)
        loss_critic_b = self._update_critic(states, goals, actions, y,
                                            self.network.critic_b, self.critic_optimizer_b)

        if self.steps % self.net_args.target_frequency == 0:
            loss_actor = self._update_actor(states, goals)

            polyak = self.net_args.polyak
            self._update_target(self.network.actor_target.variables, self.network.actor.variables, polyak)
            self._update_target(self.network.critic_target_a.variables, self.network.critic_a.variables, polyak)
            self._update_target(self.network.critic_target_a.variables, self.network.critic_b.variables, polyak)

            self.monitor.log(loss_actor, 'DDPG_loss_actor')

        self.monitor.log(loss_critic_a, 'DDPG_loss_critic_a')
        self.monitor.log(loss_critic_b, 'DDPG_loss_critic_b')
        self.steps += 1

    @tf.function
    def _update_critic(self, states, goals, actions, targets, critic, optimizer) -> tf.Tensor:
        with tf.GradientTape() as tape:
            critic_predict = critic([states, actions, goals])
            loss = tf.math.reduce_sum(scalar_loss(critic_predict, targets))

        grad = tape.gradient(loss, critic.variables)
        optimizer.apply_gradients(zip(grad, critic.variables))

        return loss

    @tf.function
    def _update_actor(self, states, goals) -> tf.Tensor:
        with tf.GradientTape() as tape:
            actions = self.network.actor([states, goals])
            critic_value = self.network.critic_a([states, actions, goals])

            loss = -tf.math.reduce_sum(critic_value)

        grad = tape.gradient(loss, self.network.actor.variables)
        self.actor_optimizer.apply_gradients(zip(grad, self.network.actor.variables))

        return loss

    @staticmethod
    @tf.function
    def _update_target(target, weights, polyak) -> None:
        for (a, b) in zip(target, weights):
            a.assign(polyak * a + (1 - polyak) * b)

    def save_checkpoint(self, folder: str = 'checkpoint', filename: str = 'checkpoint.pth.tar') -> None:
        pass

    def load_checkpoint(self, folder: str = 'checkpoint', filename: str = 'checkpoint.pth.tar') -> None:
        pass
