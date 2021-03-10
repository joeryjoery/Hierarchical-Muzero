"""
When calling Main.py, there are imports that lead tensorflow to be initialized. This file must always be completely
imported before utilizing other parts of our code pipeline as tensorflow will automatically allocate all available
VRAM. This will prevent new runs from accessing the GPUs and may result in a crash.

Additionally, this file contains a variety of monitoring flags and functions for tracking progress of agents.
"""
from abc import ABC, abstractmethod
import logging
import typing
import os

# Dynamic VRAM growth: https://github.com/tensorflow/tensorflow/issues/24496#issuecomment-464909727
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import tensorflow as tf
import numpy as np

from utils.loss_utils import support_to_scalar, safe_l2norm

# Dynamic VRAM growth: https://github.com/tensorflow/tensorflow/issues/24496#issuecomment-464909727
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)  # TODO: Set main session on the CPU. Activate Pipeline with GPU session.
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Suppress warnings from TENSORFLOW's side
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
tf.autograph.set_verbosity(1)

# Debugging flags. SET AS GLOBAL VARIABLES BEFORE RUNNING COACH.
DEBUG_MODE = False
LOG_RATE = 1
RENDER = False


class Monitor(ABC):
    """ Main interface for monitoring progress of agent """

    def __init__(self, reference):
        self.reference = reference  # Instance of a Neural Network framework to track statistics on.

    def log(self, tensor: typing.Union[tf.Tensor, float], name: str) -> None:
        """ Log a scalar annotated by the number of backpropagation steps """
        if self.reference.steps % LOG_RATE == 0:
            tf.summary.scalar(name, data=tensor, step=self.reference.steps)

    def log_distribution(self, tensor: typing.Union[tf.Tensor, np.ndarray], name: str) -> None:
        """ Log an array of scalars annotated by the number of backpropagation steps """
        if self.reference.steps % LOG_RATE == 0:
            tf.summary.histogram(name, tensor, step=self.reference.steps)

    @abstractmethod
    def log_batch(self, data_batch: typing.List) -> None:
        """ Ad-Hoc function to log specific statistic based on a batch of training-data. """


class MuZeroMonitor(Monitor):
    """
    Implementation of the Monitor class to add batch logging functionality for MuZero agents.
    Also adds functionality to log the losses of each recurrent head during unrolling/ training of the RNN.
    """

    def __init__(self, reference):
        super().__init__(reference)

    def log_recurrent_losses(self, t: int, v_loss: tf.Tensor, r_loss: tf.Tensor, pi_loss: tf.Tensor,
                             absorb: tf.Tensor, o_loss: tf.Tensor = None, *vargs) -> None:
        """ Log each prediction head loss from the MuZero RNN as a scalar (optionally includes decoder loss). """
        step = self.reference.steps
        if self.reference.steps % LOG_RATE == 0:
            tf.summary.scalar(f"r_loss_{t}", data=tf.reduce_mean(r_loss), step=step)
            tf.summary.scalar(f"v_loss_{t}", data=tf.reduce_mean(v_loss), step=step)
            tf.summary.scalar(f"pi_loss_{t}", data=tf.reduce_sum(pi_loss) / tf.reduce_sum(1 - absorb), step=step)

            if o_loss is not None:  # Decoder option.
                tf.summary.scalar(f"decode_loss_{t}", data=tf.reduce_mean(o_loss), step=step)

    def log_forward_predictions(self, unrolled_outputs, forward_predictions):
        forward_observations = np.asarray(forward_predictions)
        # Compute statistics related to auto-encoding state dynamics:
        for t, (s, v, r, pi, absorb) in enumerate(unrolled_outputs):
            k = t + 1
            stacked_obs = forward_observations[:, t, ...]

            s_enc = self.reference.neural_net.encoder.predict_on_batch(stacked_obs)
            kl_divergence = tf.keras.losses.kullback_leibler_divergence(s_enc, s)

            # Relative entropy of dynamics model and encoder.
            # Lower values indicate that the prediction model receives more stable input.
            self.log_distribution(kl_divergence, f"KL_Divergence_{k}")
            self.log(np.mean(kl_divergence), f"Mean_KLDivergence_{k}")

            # Internal entropy of the dynamics model
            s_entropy = tf.keras.losses.categorical_crossentropy(s, s)
            self.log(np.mean(s_entropy), f"mean_dynamics_entropy_{k}")

            if hasattr(self.reference.neural_net, "decoder"):
                # If available, track the performance of a neural decoder from latent to real state.
                stacked_obs_predict = self.reference.neural_net.decoder.predict_on_batch(s)
                se = (stacked_obs - stacked_obs_predict) ** 2

                self.log(np.mean(se), f"decoder_error_{k}")

    def log_value_predictions(self, out, target, k: int, prefix: str):
        self.log_distribution(out, f'{prefix}_predict_{k}')
        self.log_distribution(target, f'{prefix}_target_{k}')
        self.log(np.mean((out - target) ** 2), f"{prefix}_mse_{k}")

    def log_batch(self, data_batch: typing.List) -> None:
        """
        Log a large amount of neural network statistics based on the given batch.
        Functionality can be toggled on by specifying '--debug' as a console argument to Main.py.
        Note: toggling this functionality on will produce significantly larger tensorboard event files!

        Statistics include:
         - Priority sampling sample probabilities.
         - Loss of each recurrent head per sample as a distribution.
         - Loss discrepancy between cross-entropy and MSE for the reward/ value predictions.
         - Norm of the neural network's weights.
         - Divergence between the dynamics and encoder functions.
         - Squared error of the decoding function.
        """
        if DEBUG_MODE and self.reference.steps % LOG_RATE == 0:
            observations, actions, targets, forward_observations, sample_weight = list(zip(*data_batch))
            actions, sample_weight = np.asarray(actions), np.asarray(sample_weight)
            target_vs, target_rs, target_pis = list(map(np.asarray, zip(*targets)))

            priority = sample_weight * len(data_batch)  # Undo 1/n scaling to get priority
            self.log_distribution(priority, 'sample probability')

            # Initial inference logging
            s, pi, v = self.reference.neural_net.forward.predict_on_batch(np.asarray(observations))
            pi_loss = -np.sum(target_pis[:, 0] * np.log(pi + 1e-8), axis=-1)
            v_real = support_to_scalar(v, self.reference.net_args.support_size).ravel()

            self.log_value_predictions(v_real, target_vs[:, 0], 0, 'v')
            self.log_distribution(pi_loss, f"pi_dist_{0}")

            # Sum over one-hot-encoded actions. If this sum is zero, then there is no action --> leaf node.
            absorb_k = 1.0 - tf.reduce_sum(target_pis, axis=-1)

            # Recurrent inference logging
            collect = list()
            for k in range(actions.shape[1]):
                r, s, pi, v = self.reference.neural_net.recurrent.predict_on_batch([s, actions[:, k, :]])

                collect.append((s, v, r, pi, absorb_k[k + 1, :]))

            for t, (s, v, r, pi, absorb) in enumerate(collect):
                k = t + 1

                pi_loss = -np.sum(target_pis[:, k] * np.log(pi + 1e-8), axis=-1)
                self.log_distribution(pi_loss, f"pi_dist_{k}")

                v_real = support_to_scalar(v, self.reference.net_args.support_size).ravel()
                r_real = support_to_scalar(r, self.reference.net_args.support_size).ravel()

                self.log_value_predictions(v_real, target_vs[:, k], k, 'v')
                self.log_value_predictions(r_real, target_rs[:, k], k, 'r')

            l2_norm = tf.reduce_sum([safe_l2norm(x) for x in self.reference.get_variables()])
            self.log(l2_norm, "l2 norm")

            # Option to track statistical properties of the dynamics model.
            if self.reference.net_args.dynamics_penalty > 0:
                self.log_forward_predictions(unrolled_outputs=collect, forward_predictions=forward_observations)


class ContinuousMuZeroMonitor(MuZeroMonitor):
    """
    Override MuZero Monitor class to log implementation specific data from the Continuous version of MuZero.
    """

    def log_recurrent_losses(self, t: int, v_loss: tf.Tensor, r_loss: tf.Tensor, param_loss: tf.Tensor,
                             absorb: tf.Tensor, *vargs) -> None:
        """ Log each prediction head loss from the MuZero RNN as a scalar (optionally includes decoder loss). """
        step = self.reference.steps
        if self.reference.steps % LOG_RATE == 0:
            tf.summary.scalar(f"r_loss_{t}", data=tf.reduce_mean(r_loss), step=step)
            tf.summary.scalar(f"v_loss_{t}", data=tf.reduce_mean(v_loss), step=step)
            tf.summary.scalar(f"pi_loss_{t}", data=tf.reduce_sum(param_loss) / tf.reduce_sum(1 - absorb), step=step)

            self.log_param_loss_components(t, absorb, vargs[0], vargs[1])

    def log_param_loss_components(self, t: int, absorb, param_cross_entropy, param_entropy):
        tf.summary.scalar(f"pi_cross_entropy_{t}", data=tf.reduce_sum(param_cross_entropy) / tf.reduce_sum(1 - absorb),
                          step=self.reference.steps)
        tf.summary.scalar(f"pi_entropy_{t}", data=tf.reduce_sum(param_entropy) / tf.reduce_sum(1 - absorb),
                          step=self.reference.steps)

    def log_parameter_predictions(self, params, target_params, action_support, k):
        dist = self.reference.distribution(params[..., 0], params[..., 1])  # TODO: Multidimensional
        self.log_distribution(dist.mean(), f'mean_{k}')
        self.log_distribution(dist.variance(), f'variance_{k}')
        self.log_distribution(params[..., 0], f'scale_a_{k}')
        self.log_distribution(params[..., 1], f'scale_b_{k}')

        target_entropy = -np.sum(target_params * np.log(target_params + 1e-8), axis=-1)
        self.log_distribution(target_entropy, f'target_entropy_{k}')
        self.log(np.mean(target_entropy), f'avg_target_entropy_{k}')

        expected_action = np.sum(target_params[..., np.newaxis] * action_support, axis=-2)
        action_deviance = np.std(target_params[..., np.newaxis] * action_support, axis=-2)

        self.log_distribution(expected_action, f'expected_action_{k}')
        self.log(np.mean(action_deviance), f'action_deviance_{k}')

    def log_batch(self, data_batch: typing.List) -> None:
        """ TODO: Update Doc
        Log a large amount of neural network statistics based on the given batch.
        Functionality can be toggled on by specifying '--debug' as a console argument to Main.py.
        Note: toggling this functionality on will produce significantly larger tensorboard event files!

        Statistics include:
         - Priority sampling sample probabilities.
         - Loss of each recurrent head per sample as a distribution.
         - Loss discrepancy between cross-entropy and MSE for the reward/ value predictions.
         - Norm of the neural network's weights.
         - Divergence between the dynamics and encoder functions.
         - Squared error of the decoding function.
        """
        if DEBUG_MODE and self.reference.steps % LOG_RATE == 0:
            observations, actions, targets, forward_observations, action_support, sample_weight = list(zip(*data_batch))
            actions, sample_weight = np.asarray(actions), np.asarray(sample_weight)
            target_vs, target_rs, target_params = list(map(np.asarray, zip(*targets)))
            action_support = np.asarray(action_support)

            priority = sample_weight * len(data_batch)  # Undoes 1/n scaling to get raw priorities
            self.log_distribution(priority, 'sample probability')

            # Initial inference logging
            s, param, v = self.reference.neural_net.forward.predict_on_batch(np.asarray(observations))
            v_real = support_to_scalar(v, self.reference.net_args.support_size).ravel()

            self.log_value_predictions(v_real, target_vs[:, 0], 0, 'v')
            self.log_parameter_predictions(param, target_params[:, 0], action_support[:, 0], 0)

            # Sum over one-hot-encoded actions. If this sum is zero, then there is no action --> leaf node.
            absorb_k = 1.0 - tf.reduce_sum(target_params, axis=-1)

            # Recurrent inference logging
            collect = list()
            for k in range(actions.shape[1]):
                r, s, param, v = self.reference.neural_net.recurrent.predict_on_batch([s, actions[:, k, :]])

                collect.append((s, v, r, param, absorb_k[k + 1, :]))

            for t, (s, v, r, param, absorb) in enumerate(collect):
                k = t + 1

                self.log_parameter_predictions(param, target_params[:, k], action_support[:, k], k)

                v_real = support_to_scalar(v, self.reference.net_args.support_size).ravel()
                r_real = support_to_scalar(r, self.reference.net_args.support_size).ravel()

                self.log_value_predictions(v_real, target_vs[:, k], k, 'v')
                self.log_value_predictions(r_real, target_rs[:, k], k, 'r')

            l2_norm = tf.reduce_sum([safe_l2norm(x) for x in self.reference.get_variables()])
            self.log(l2_norm, "l2 norm")

            # Option to track statistical properties of the dynamics model.
            if self.reference.net_args.dynamics_penalty > 0:
                self.log_forward_predictions(unrolled_outputs=collect, forward_predictions=forward_observations)


class AlphaZeroMonitor(Monitor):
    """ Implementation of the Monitor class to add batch logging functionality for AlphaZero agents. """

    def __init__(self, reference):
        super().__init__(reference)

    def log_batch(self, data_batch: typing.List) -> None:
        """
        Log a large amount of neural network statistics based on the given batch.
        Functionality can be toggled on by specifying '--debug' as a console argument to Main.py.
        Note: toggling this functionality on will produce significantly larger tensorboard event files!

        Statistics include:
         - Priority sampling sample probabilities.
         - Values of each target/ prediction for the data batch.
         - Loss discrepancy between cross-entropy and MSE for the reward/ value predictions.
        """
        if DEBUG_MODE and self.reference.steps % LOG_RATE == 0:
            observations, targets, sample_weight = list(zip(*data_batch))
            target_pis, target_vs = list(map(np.asarray, zip(*targets)))
            observations = np.asarray(observations)

            priority = sample_weight * len(data_batch)  # Undo 1/n scaling to get priority
            tf.summary.histogram(f"sample probability", data=priority, step=self.reference.steps)

            pis, vs = self.reference.neural_net.model.predict_on_batch(observations)
            v_reals = support_to_scalar(vs, self.reference.net_args.support_size).ravel()  # as scalars

            tf.summary.histogram(f"v_targets", data=target_vs, step=self.reference.steps)
            tf.summary.histogram(f"v_predict", data=v_reals, step=self.reference.steps)

            mse = np.mean((v_reals - target_vs) ** 2)
            tf.summary.scalar("v_mse", data=mse, step=self.reference.steps)


class HierarchicalMuZeroMonitor(Monitor):

    def log_goal_statistics(self, training_data: typing.List) -> None:

        goal_success = list()
        goal_length = list()
        distances = list()
        for episode in training_data:
            len_episode = len(episode)
            num_goals = len(episode.goals)

            lengths = np.asarray(episode.goal_indices + [len_episode])[1:] - np.asarray(episode.goal_indices)

            achieved = [int(episode.goals[i - 1].achieved) for i in episode.goal_indices[1:] + [0]
                        if episode.goals[i - 1].subgoal_testing]  # [0] -> [-1]
            distance = [np.linalg.norm(episode.goals[i - 1].goal - episode.next_observations[i - 1])
                        for i in episode.goal_indices[1:] + [0]]  # [0] -> [-1]

            success_ratio = np.sum(achieved) / len(achieved)

            distances.append(np.mean(distance))
            goal_success.append(success_ratio)
            goal_length.append(np.mean(lengths))

            # print(success_ratio, lengths, np.mean(lengths), achieved, distance)  # DEBUGGING

        self.log(np.mean(distances), 'avg_goal_separation')
        self.log(np.mean(goal_success), 'avg_goal_success')
        self.log(np.mean(goal_length), 'avg_goal_length')

        self.log_distribution(distances, 'goal_separation')
        self.log_distribution(np.asarray(goal_success), 'goal_success')
        self.log_distribution(np.asarray(goal_length), 'goal_length')

        if DEBUG_MODE:
            pass  # Log additional statistics. TODO

    def log_batch(self, data_batch: typing.List) -> None:
        goal_examples, act_examples = data_batch

        # Network specific logging.
        self.reference.goal_net.monitor.log_batch(goal_examples)
        self.reference.action_net.monitor.log_batch(act_examples)

        # Joint policy logging.
        # TODO


class ModelFreeMonitor(Monitor):

    def log_batch(self, data_batch: typing.List) -> None:
        pass  # TODO
