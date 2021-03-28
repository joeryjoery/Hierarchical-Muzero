"""
Defines default MLP neural networks for both AlphaZero and MuZero that can be used on simple classical control
environments, such as those from OpenAI Gym.
"""

import sys

import numpy as np

from keras.layers import Dense, Input, Reshape, Concatenate, Activation, Softmax, Lambda
from keras.models import Model
from keras.optimizers import Adam

from utils.network_utils import Crafter, MinMaxScaler

sys.path.append('../..')


class AlphaZeroGymNetwork:

    def __init__(self, game, args):
        # Network arguments
        self.x, self.y, self.planes = game.getDimensions()

        self.action_size = game.getActionSize()
        self.args = args
        self.crafter = Crafter(args)

        # s: batch_size x time x state_x x state_y
        self.observation_history = Input(shape=(self.x, self.y, self.planes * self.args.observation_length))
        # a: one hot encoded vector of shape batch_size x (state_x * state_y)
        self.action_tensor = Input(shape=(self.action_size,))

        observations = Reshape((self.x * self.y * self.planes * self.args.observation_length,))(
            self.observation_history)

        self.pi, self.v = self.build_predictor(observations)

        self.model = Model(inputs=self.observation_history, outputs=[self.pi, self.v])

        opt = Adam(args.optimizer.lr_init)
        if self.args.support_size > 0:
            self.model.compile(loss=['categorical_crossentropy'] * 2, optimizer=opt)
        else:
            self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=opt)

    def build_predictor(self, observations):
        fc_sequence = self.crafter.dense_sequence(self.args.num_dense, observations)

        pi = Dense(self.action_size, activation='softmax', name='pi')(fc_sequence)
        v = Dense(1, activation='linear', name='v')(fc_sequence) \
            if self.args.support_size == 0 else \
            Dense(self.args.support_size * 2 + 1, activation='softmax', name='v')(fc_sequence)

        return pi, v


class MuZeroGymNetwork:

    def __init__(self, game, args):
        # Network arguments
        self.x, self.y, self.planes = game.getDimensions()
        self.latents = args.latent_depth
        self.action_size = game.getActionSize()
        self.args = args
        self.crafter = Crafter(args)

        # s: batch_size x time x state_x x state_y
        self.observation_history = Input(shape=(self.x, self.y, self.planes))
        # a: one hot encoded vector of shape batch_size x (state_x * state_y)
        self.action_tensor = Input(shape=(self.action_size,))
        # s': batch_size  x board_x x board_y x 1
        self.latent_state = Input(shape=(self.latents, 1))

        observations = Reshape((self.x * self.y * self.planes,))(self.observation_history)
        latent_state = Reshape((self.latents,))(self.latent_state)

        # Build tensorflow computation graph
        self.s = self.build_encoder(observations)
        self.r, self.s_next = self.build_dynamics(latent_state, self.action_tensor)
        self.pi, self.v = self.build_predictor(latent_state)

        self.encoder = Model(inputs=self.observation_history, outputs=self.s, name="r")
        self.dynamics = Model(inputs=[self.latent_state, self.action_tensor], outputs=[self.r, self.s_next], name='d')
        self.predictor = Model(inputs=self.latent_state, outputs=[self.pi, self.v], name='p')

        self.forward = Model(inputs=self.observation_history, outputs=[self.s, *self.predictor(self.s)], name='initial')
        self.recurrent = Model(inputs=[self.latent_state, self.action_tensor],
                               outputs=[self.r, self.s_next, *self.predictor(self.s_next)], name='recurrent')

        # Decoder functionality.
        self.decoded_observations = self.build_decoder(latent_state)
        self.decoder = Model(inputs=self.latent_state, outputs=self.decoded_observations, name='decoder')

    def build_encoder(self, observations):
        fc_sequence = self.crafter.dense_sequence(self.args.num_dense, observations)

        latent_state = Dense(self.latents, activation='linear', name='s_0')(fc_sequence)
        latent_state = Activation('tanh')(latent_state) if self.latents <= 3 else MinMaxScaler()(latent_state)
        latent_state = Reshape((self.latents, 1))(latent_state)

        return latent_state  # 2-dimensional 1-time step latent state. (Encodes history of images into one state).

    def build_dynamics(self, encoded_state, action_plane):
        stacked = Concatenate()([encoded_state, action_plane])
        fc_sequence = self.crafter.dense_sequence(self.args.num_dense, stacked)

        latent_state = Dense(self.latents, activation='linear', name='s_next')(fc_sequence)
        latent_state = Activation('tanh')(latent_state) if self.latents <= 3 else MinMaxScaler()(latent_state)
        latent_state = Reshape((self.latents, 1))(latent_state)

        r = Dense(1, activation='linear', name='r')(fc_sequence) \
            if self.args.support_size == 0 else \
            Dense(self.args.support_size * 2 + 1, activation='softmax', name='r')(fc_sequence)

        return r, latent_state

    def build_predictor(self, latent_state):
        fc_sequence = self.crafter.dense_sequence(self.args.num_dense, latent_state)

        pi = Dense(self.action_size, activation='softmax', name='pi')(fc_sequence)
        v = Dense(1, activation='linear', name='v')(fc_sequence) \
            if self.args.support_size == 0 else \
            Dense(self.args.support_size * 2 + 1, activation='softmax', name='v')(fc_sequence)

        return pi, v

    def build_decoder(self, latent_state):
        fc_sequence = self.crafter.dense_sequence(self.args.num_dense, latent_state)

        out = Dense(self.x * self.y * self.planes, name='o_k')(fc_sequence)
        o = Reshape((self.x, self.y, self.planes))(out)
        return o


class ContinuousMuZeroGymNetwork(MuZeroGymNetwork):
    # The only thing that needs to be altered for a continuous network is that the predictor function outputs two
    # predictions for each action, one being the mean of a parameterized Gaussian and the second it standard deviation.

    def build_predictor(self, latent_state):
        fc_sequence = self.crafter.dense_sequence(self.args.num_dense, latent_state)

        n = np.prod(self.action_size)
        params = Dense(n * 2, activation='softplus', name='params')(fc_sequence)
        params = Reshape((n, 2))(params)

        v = Dense(1, activation='linear', name='v')(fc_sequence) \
            if self.args.support_size == 0 else \
            Dense(self.args.support_size * 2 + 1, activation='softmax', name='v')(fc_sequence)

        return params, v


class GoalConditionedMuZeroGymNetwork:

    def __init__(self, game, args, goal_sampling=False):
        # Network arguments
        self.x, self.y, self.planes = game.getDimensions()
        self.latents = args.latent_depth
        self.action_size = game.getActionSize() if not goal_sampling else self.get_goal_space(args)
        self.goal_size = self.get_goal_space(args)  # TODO: Add tensor for goal age? (useful for learning?)
        self.args = args
        self.crafter = Crafter(args)

        # s: batch_size x time x state_x x state_y
        self.observation_history = Input(shape=(self.x, self.y, self.planes))
        # a: one hot encoded vector of shape batch_size x (state_x * state_y)
        self.action_tensor = Input(shape=(self.action_size,))
        self.goal_tensor = Input(shape=(self.x, self.y, self.goal_size))  # TODO: add goal+tensor to encoder?
        # s': batch_size  x board_x x board_y x 1
        self.latent_state = Input(shape=(self.latents, 1))

        observations = Reshape((self.x * self.y * self.planes,))(self.observation_history)
        latent_state = Reshape((self.latents,))(self.latent_state)
        goal = Reshape((self.goal_size,))(self.goal_tensor)

        # Build tensorflow computation graph
        self.s = self.build_encoder(observations)
        self.r, self.s_next = self.build_dynamics(latent_state, self.action_tensor)
        self.pi, self.v = self.build_predictor(latent_state)

        self.encoder = Model(inputs=[self.observation_history, self.goal_tensor], outputs=self.s, name="r")
        self.dynamics = Model(inputs=[self.latent_state, self.action_tensor], outputs=[self.r, self.s_next], name='d')
        self.predictor = Model(inputs=self.latent_state, outputs=[self.pi, self.v], name='p')

        if goal_sampling:
            self.forward = Model(inputs=self.observation_history, outputs=[self.s, *self.predictor([self.s])], name='initial')
        else:
            self.forward = Model(inputs=[self.observation_history, self.goal_tensor],
                                 outputs=[self.s, *self.predictor([self.s])], name='initial')
        self.recurrent = Model(inputs=[self.latent_state, self.action_tensor],
                               outputs=[self.r, self.s_next, *self.predictor(self.s_next)], name='recurrent')

    def get_goal_space(self, args):
        if args.goal_space.latent:
            return self.latents
        else:
            return self.planes  # Not image-gym friendly.

    def build_encoder(self, observations):
        fc_sequence = self.crafter.dense_sequence(self.args.num_dense, observations)

        latent_state = Dense(self.latents, activation='linear', name='s_0')(fc_sequence)
        latent_state = Activation('tanh')(latent_state) if self.latents <= 3 else MinMaxScaler()(latent_state)
        latent_state = Reshape((self.latents, 1))(latent_state)

        return latent_state  # 2-dimensional 1-time step latent state. (Encodes history of images into one state).

    def build_dynamics(self, encoded_state, action_plane):
        stacked = Concatenate()([encoded_state, action_plane])
        fc_sequence = self.crafter.dense_sequence(self.args.num_dense, stacked)

        latent_state = Dense(self.latents, activation='linear', name='s_next')(fc_sequence)
        latent_state = Activation('tanh')(latent_state) if self.latents <= 3 else MinMaxScaler()(latent_state)
        latent_state = Reshape((self.latents, 1))(latent_state)

        r = Dense(1, activation='linear', name='r')(fc_sequence) \
            if self.args.support_size == 0 else \
            Dense(self.args.support_size * 2 + 1, activation='softmax', name='r')(fc_sequence)

        return r, latent_state

    def build_predictor(self, latent_state):
        fc_sequence = self.crafter.dense_sequence(self.args.num_dense, latent_state)

        pi = Dense(self.action_size, activation='softmax', name='pi')(fc_sequence)
        v = Dense(1, activation='linear', name='v')(fc_sequence) \
            if self.args.support_size == 0 else \
            Dense(self.args.support_size * 2 + 1, activation='softmax', name='v')(fc_sequence)

        return pi, v


class GoalConditionedContinuousMuZeroGymNetwork(GoalConditionedMuZeroGymNetwork):

    def build_predictor(self, latent_state):
        fc_sequence = self.crafter.dense_sequence(self.args.num_dense, latent_state)

        n = np.prod(self.action_size)
        params = Dense(n * 2, activation='softplus', name='params')(fc_sequence)
        params = Reshape((n, 2))(params)

        v = Dense(1, activation='linear', name='v')(fc_sequence) \
            if self.args.support_size == 0 else \
            Dense(self.args.support_size * 2 + 1, activation='softmax', name='v')(fc_sequence)

        return params, v


class DQNGymNetwork:

    def __init__(self, game, args):
        # Network arguments
        self.x, self.y, self.planes = game.getDimensions()
        self.latents = args.latent_depth
        self.action_size = game.getActionSize()
        self.goal_size = self.get_goal_space(args)
        self.args = args
        self.crafter = Crafter(args)

        self.observations = Input(shape=(self.x, self.y, self.planes))
        self.goal = Input(shape=(self.x, self.y, self.goal_size))
        self.model_in = [self.observations, self.goal]

        observations = Reshape((self.x * self.y * self.planes,))(self.observations)
        goal = Reshape((self.x * self.y * self.goal_size,))(self.goal)

        self.out_tensor = self.build_model([observations, goal])
        self.out_tensor_copy = self.build_model([observations, goal])

        self.model = Model(inputs=self.model_in, outputs=self.out_tensor, name='UVFA_DQN')
        self.target = Model(inputs=self.model_in, outputs=self.out_tensor_copy, name='UVFA_DQN_target')

    def build_model(self, inputs):
        concat = Concatenate()(inputs)
        fc_sequence = self.crafter.dense_sequence(self.args.actor_size, concat)

        qs = Dense(self.action_size, activation='linear', name='q')(fc_sequence)

        # Scale to [-horizon, 0]. Input to sigmoid is shifted for an optimistic initialiation (near 0)
        qs = Lambda(lambda x: Activation('sigmoid')(x - self.args.actor_bound) * -self.args.actor_bound)(qs)

        return qs

    def get_goal_space(self, args):
        if args.goal_space.latent:
            return self.latents
        else:
            return self.planes  # Not image-gym friendly.


class DDPGGymNetwork:

    def __init__(self, game, args, goal_sampling=False):
        # Network arguments
        self.x, self.y, self.planes = game.getDimensions()
        self.latents = args.latent_depth
        self.action_size = game.getActionSize() if not goal_sampling else self.get_goal_space(args)
        self.goal_size = self.get_goal_space(args)
        self.args = args
        self.crafter = Crafter(args)

        self.observations = Input(shape=(self.x, self.y, self.planes))
        self.action_tensor = Input(shape=(self.action_size,))
        self.goal = Input(shape=(self.x, self.y, self.goal_size))

        observations = Reshape((self.x * self.y * self.planes,))(self.observations)  # TODO incorporate
        goal = Reshape((self.x * self.y * self.goal_size,))(self.goal)

        self.actor_in = [self.observations, self.goal]
        self.critic_in = [self.observations, self.action_tensor, self.goal]

        self.actor_out = self.build_actor()
        self.actor_target = self.build_actor()
        self.actor = Model(inputs=[self.observations, self.goal], outputs=self.actor_out, name='UVFA_DDPG_actor')
        self.actor_target = Model(inputs=[self.observations, self.goal], outputs=self.actor_target, name='UVFA_DDPG_target')

        self.critic_out_a = self.build_critic()
        self.critic_target_a = self.build_critic()
        self.critic_a = Model(inputs=self.critic_in, outputs=self.critic_out_a, name='UVFA_DDPG_critic_a')
        self.critic_target_a = Model(inputs=self.critic_in, outputs=self.critic_target_a, name='UVFA_DDPG_critic_target_a')

        self.critic_out_b = self.build_critic()
        self.critic_target_b = self.build_critic()
        self.critic_b = Model(inputs=self.critic_in, outputs=self.critic_out_b, name='UVFA_DDPG_critic_b')
        self.critic_target_b = Model(inputs=self.critic_in, outputs=self.critic_target_b, name='UVFA_DDPG_critic_target_b')

    def build_actor(self):
        concat = Concatenate()(self.actor_in)

        actor_seq = self.crafter.dense_sequence(self.args.actor_size, concat)
        mu = Dense(self.action_size, activation='linear', name='mu')(actor_seq)

        return mu

    def build_critic(self):
        concat = Concatenate()(self.critic_in)

        critic_seq = self.crafter.dense_sequence(self.args.critic_size, concat)
        qs = Dense(1, activation='linear', name='q')(critic_seq)

        return qs

    def get_goal_space(self, args):
        if args.goal_space.latent:
            return self.latents
        else:
            return self.planes  # Not image-gym friendly.
