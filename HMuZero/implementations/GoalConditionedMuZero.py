import typing
from abc import ABC

import numpy as np
from tensorflow import GradientTape

from MuZero.implementations.DefaultMuZero import DefaultMuZero
from MuZero.implementations.ContinuousMuZero import ContinuousMuZero
from utils import DotDict
from utils.loss_utils import support_to_scalar, scalar_to_support, cast_to_tensor
from utils.HierarchicalUtils import GoalState


class IGoalConditionedMuZero(ABC):
    current_goal: GoalState = None

    def set_goal(self, goal: GoalState) -> None:
        self.current_goal = goal


class GCDefaultMuZero(DefaultMuZero, IGoalConditionedMuZero):

    def __init__(self, game, net_args: DotDict, architecture: str, goal_sampling: bool = False) -> None:
        if goal_sampling:
            super().__init__(game, net_args, f'Goal_{architecture}')
        else:
            super().__init__(game, net_args, f'GC_{architecture}')

        self.goal_sampling = goal_sampling

        # Set default goal to zeros ==> defaults to no goal/ the canonical algorithm.
        self.set_goal(GoalState.empty(np.zeros((1, 1, self.neural_net.get_goal_space(net_args)))))

    # TODO: Loss function with goals

    def train(self, examples: typing.List) -> float:
        # Unpack and transform data for loss computation.
        observations, actions, goals, targets, forward_observations, sample_weight = list(zip(*examples))

        forward_observations = np.asarray(forward_observations)

        actions, sample_weight, goals = np.asarray(actions), np.asarray(sample_weight), np.asarray(goals)

        # Unpack and encode targets. Value target shapes are of the form [time, batch_size, categories]
        target_vs, target_rs, target_pis = list(map(np.asarray, zip(*targets)))

        target_vs = np.asarray([scalar_to_support(target_vs[:, t], self.net_args.support_size)
                                for t in range(target_vs.shape[-1])])
        target_rs = np.asarray([scalar_to_support(target_rs[:, t], self.net_args.support_size)
                                for t in range(target_rs.shape[-1])])
        target_pis = np.swapaxes(target_pis, 0, 1)

        # Pack formatted inputs as tensors.
        data = [cast_to_tensor(x) for x in [observations, actions, goals, target_vs, target_rs,
                                            target_pis, forward_observations, sample_weight]]

        # Track the gradient through unrolling and loss computation and perform an optimization step.
        with GradientTape() as tape:
            loss, step_losses = self.loss_function(*data)

        grads = tape.gradient(loss, self.get_variables())
        self.optimizer.apply_gradients(zip(grads, self.get_variables()), name=f'MuZeroDefault_{self.architecture}')

        # Logging
        self.monitor.log(loss / len(examples), "total loss")
        for k, step_loss in enumerate(step_losses):
            self.monitor.log_recurrent_losses(k, *step_loss)

        self.steps += 1

    def initial_inference(self, observations: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, float]:
        # Pad batch dimension
        observations = observations[np.newaxis, ...]
        g_plane = self.current_goal.goal[np.newaxis, ...]

        s_0, pi, v = self.neural_net.forward.predict([observations, g_plane])

        # Cast bins to scalar
        v_real = support_to_scalar(v, self.net_args.support_size)

        return s_0[0], pi[0], np.ndarray.item(v_real)

    def recurrent_inference(self, latent_state: np.ndarray, action: int) -> typing.Tuple[float, np.ndarray,
                                                                                         np.ndarray, float]:
        # One hot encode integer actions.
        a_plane = np.zeros(self.action_size)
        a_plane[action] = 1

        # Pad batch dimension
        latent_state = latent_state[np.newaxis, ...]
        a_plane = a_plane[np.newaxis, ...]
        g_plane = self.current_goal.goal[np.newaxis, ...]

        r, s_next, pi, v = self.neural_net.recurrent.predict([latent_state, a_plane])  # TODO: include goal??

        # Cast bins to scalar
        v_real = support_to_scalar(v, self.net_args.support_size)
        r_real = support_to_scalar(r, self.net_args.support_size)

        return np.ndarray.item(r_real), s_next[0], pi[0], np.ndarray.item(v_real)


class GCContinuousMuZero(ContinuousMuZero, IGoalConditionedMuZero):

    def __init__(self, game, net_args: DotDict, architecture: str, goal_sampling: bool = False) -> None:
        if goal_sampling:
            super().__init__(game, net_args, f'Goal_{architecture}')
        else:
            super().__init__(game, net_args, f'GC_{architecture}')

        self.goal_sampling = goal_sampling

        # Set default goal to zeros ==> defaults to no goal/ the canonical algorithm.
        self.set_goal(GoalState.empty(np.zeros((1, 1, self.neural_net.get_goal_space(net_args)))))

    # TODO: Loss function with goals

    def train(self, examples: typing.List) -> float:
        if self.goal_sampling:
            return super().train(examples)

        # Unpack and transform data for loss computation.
        observations, actions, goals, targets, forward_observations, action_support, sample_weight = list(
            zip(*examples))

        forward_observations = np.asarray(forward_observations)

        actions, sample_weight, goals = np.asarray(actions), np.asarray(sample_weight), np.asarray(goals)
        action_support = np.asarray(action_support)

        # Unpack and encode targets. Value target shapes are of the form [time, batch_size, categories]
        target_vs, target_rs, target_params = list(map(np.asarray, zip(*targets)))

        print(target_vs.shape, target_vs)

        target_vs = np.asarray([scalar_to_support(target_vs[:, t], self.net_args.support_size)
                                for t in range(target_vs.shape[-1])])
        target_rs = np.asarray([scalar_to_support(target_rs[:, t], self.net_args.support_size)
                                for t in range(target_rs.shape[-1])])
        target_params = np.swapaxes(target_params, 0, 1)[..., np.newaxis]  # Pad also with action-space dimension.
        action_support = np.swapaxes(action_support, 0, 1)

        # Pack formatted inputs as tensors.
        data = [cast_to_tensor(x) for x in [observations, actions, goals, target_vs, target_rs,
                                            target_params, forward_observations, action_support, sample_weight]]

        # Track the gradient through unrolling and loss computation and perform an optimization step.
        with GradientTape() as tape:
            loss, step_losses = self.loss_function(*data)

        grads = tape.gradient(loss, self.get_variables())
        self.optimizer.apply_gradients(zip(grads, self.get_variables()), name=f'MuZeroContinuous_{self.architecture}')

        # Logging
        self.monitor.log(loss / len(examples), "total loss")
        for k, step_loss in enumerate(step_losses):
            self.monitor.log_recurrent_losses(k, *step_loss)

        self.steps += 1

    def initial_inference(self, observations: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, float]:
        if self.goal_sampling:
            return super().initial_inference(observations)

        observations = observations[np.newaxis, ...]
        g_plane = self.current_goal.goal[np.newaxis, ...]

        s_0, pi, v = self.neural_net.forward.predict([observations, g_plane])

        # Cast bins to scalar
        v_real = support_to_scalar(v, self.net_args.support_size)

        return s_0[0], pi[0], np.ndarray.item(v_real)

    def recurrent_inference(self, latent_state: np.ndarray, action: np.ndarray) -> typing.Tuple[float, np.ndarray,
                                                                                         np.ndarray, float]:
        if self.goal_sampling:
            return super().recurrent_inference(latent_state, action)

        # Pad batch dimension
        latent_state = latent_state[np.newaxis, ...]
        a_plane = np.asarray(action)[np.newaxis, ...]
        g_plane = self.current_goal.goal[np.newaxis, ...]

        r, s_next, pi, v = self.neural_net.recurrent.predict([latent_state, a_plane])  # TODO: include goal??

        # Cast bins to scalar
        v_real = support_to_scalar(v, self.net_args.support_size)
        r_real = support_to_scalar(r, self.net_args.support_size)

        return np.ndarray.item(r_real), s_next[0], pi[0], np.ndarray.item(v_real)
