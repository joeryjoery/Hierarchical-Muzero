"""
In this python file we modified the class to enable generation of multiple
neural architectures by modifying the argument dictionary.

For the details of the neural architectures, we refer to our report.
:see: main_experimenter.py
"""

import sys

from keras.models import *
from keras.layers import *

from utils.network_utils import MinMaxScaler

sys.path.append('../..')


class AtariNNet:

    def __init__(self, game, args):
        # Network arguments
        self.board_x, self.board_y, self.planes = game.getDimensions()
        self.latent_x, self.latent_y = (6, 6)
        self.action_size = game.getActionSize()
        self.args = args

        # s: batch_size x time x state_x x state_y
        self.observation_history = Input(shape=(self.board_x, self.board_y, self.planes * self.args.observation_length))
        # a: one hot encoded vector of shape batch_size x (state_x * state_y)
        self.action_plane = Input(shape=(self.latent_x, self.latent_y))
        # s': batch_size  x board_x x board_y x 1
        self.latent_state = Input(shape=(self.latent_x, self.latent_y))

        action_plane = Reshape((self.latent_x, self.latent_y, 1))(self.action_plane)
        latent_state = Reshape((self.latent_x, self.latent_y, 1))(self.latent_state)

        self.s = self.encoder(self.observation_history)
        self.r, self.s_next = self.dynamics(latent_state, action_plane)

        self.pi, self.v = self.predictor(latent_state)

        self.encoder = Model(inputs=self.observation_history, outputs=self.s)
        self.dynamics = Model(inputs=[self.latent_state, self.action_plane], outputs=[self.r, self.s_next])
        self.predictor = Model(inputs=self.latent_state, outputs=[self.pi, self.v])

        self.pi2, self.v2 = self.predictor(self.s_next)
        self.recurrent = Model(inputs=[self.latent_state, self.action_plane], outputs=[self.r, self.s_next, self.pi2, self.v2])

    def build_model(self, tensor_in):
        def conv_block(n, x):  # Recursively builds a convolutional tower of height n.
            if n > 0:
                return conv_block(n - 1, Activation('relu')(BatchNormalization()(Conv2D(
                    self.args.num_channels, 3, padding='same', use_bias=False)(x))))
            return x

        def dense_sequence(n, x):  # Recursively builds a Fully Connected sequence of length n.
            if n > 0:
                return dense_sequence(n - 1, Dropout(self.args.dropout)(Activation('relu')(
                    Dense(self.args.size_dense)(x))))
            return x

        conv_block = conv_block(self.args.num_towers, tensor_in)
        flattened = Flatten()(conv_block)
        fc_sequence = dense_sequence(self.args.len_dense, flattened)
        return fc_sequence

    def encoder(self, observations):

        downsampled = Activation('relu')(BatchNormalization()(Conv2D(64, 3, 2)(observations)))
        downsampled = Activation('relu')(BatchNormalization()(Conv2D(128, 3, 2)(downsampled)))
        downsampled = AveragePooling2D(3, 2)(downsampled)
        downsampled = AveragePooling2D(3, 2)(downsampled)

        out_tensor = self.build_model(downsampled)

        s_fc_latent = Dense(self.latent_x * self.latent_y, activation='linear', name='s_0')(out_tensor)
        latent_state = MinMaxScaler(safe=True)(s_fc_latent)
        latent_state = Reshape((self.latent_x, self.latent_y, 1))(latent_state)

        return latent_state  # 2-dimensional 1-time step latent state. (Encodes history of images into one state).

    def dynamics(self, encoded_state, action_plane):
        stacked = Concatenate(axis=-1)([encoded_state, action_plane])
        reshaped = Reshape((self.latent_x, self.latent_y, -1))(stacked)
        out_tensor = self.build_model(reshaped)

        s_fc_latent = Dense(self.latent_x * self.latent_y, activation='linear', name='s_next')(out_tensor)
        latent_state = Reshape((self.latent_x, self.latent_y, 1))(s_fc_latent)
        latent_state = MinMaxScaler(safe=True)(latent_state)

        r = Dense(1, activation='linear', name='r')(out_tensor) \
            if self.args.support_size == 0 else \
            Dense(self.args.support_size * 2 + 1, activation='softmax', name='r')(out_tensor)

        return r, latent_state

    def predictor(self, latent_state):
        out_tensor = self.build_model(latent_state)

        pi = Dense(self.action_size, activation='softmax', name='pi')(out_tensor)
        v = Dense(1, activation='tanh', name='v')(out_tensor) \
            if self.args.support_size == 0 else \
            Dense(self.args.support_size * 2 + 1, activation='softmax', name='v')(out_tensor)

        return pi, v


if __name__ == "__main__":
    
    from Games.atari.AtariGame import AtariGame as Game
    game = Game("PongNoFrameskip-v4")
    nnet = AtariNNet(game, {
        "observation_length": 10,
        "num_channels": 8,
        "size_dense": 32,
        "dropout": 0.5,
        "num_towers": 1,
        "len_dense": 2,
        "support_size": 300 
    })