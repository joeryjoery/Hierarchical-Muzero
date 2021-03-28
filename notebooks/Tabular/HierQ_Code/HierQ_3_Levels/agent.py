import pickle as cpickle

import numpy as np


def rand_argmax(arr, goal=None):
    max_ind = np.flatnonzero(arr == arr.max())

    g = np.flatnonzero(max_ind == goal)
    if len(g):
        return max_ind[np.random.choice(g)]
    return np.random.choice(max_ind)


class Agent:

    def __init__(self, num_squares, num_actions, time_limits, FLAGS, gamma=1, lr=1, epsilon=0.25):
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon

        if FLAGS.retrain:
            # If retrain, reset critic lookup table for each agent layer.  Initialize to a negative value.
            self.critic_lay2 = np.ones((num_squares, num_squares, num_squares)) * -num_squares
            self.critic_lay1 = np.ones((num_squares, num_squares, num_squares)) * -time_limits[0] * time_limits[1]
            self.critic_lay0 = np.ones((num_squares, num_squares, num_actions)) * -time_limits[0]
        else:
            # Load critic lookup table
            self.critic_lay2 = cpickle.load(open("critic_lay2_table.p", "rb"))
            self.critic_lay1 = cpickle.load(open("critic_lay1_table.p", "rb"))
            self.critic_lay0 = cpickle.load(open("critic_lay0_table.p", "rb"))
        self.tables = [self.critic_lay0, self.critic_lay1, self.critic_lay2]

    def get_action(self, s, greedy, lvl, goal=None) -> int:
        greedy = int(lvl > 0 or greedy or np.random.random_sample() > self.epsilon)
        return rand_argmax(self.tables[lvl][goal][s] * greedy, goal=goal if lvl else None)

    def update_critic(self, lvl, s, a, s_next, goals):
        # agent.update_critic(0, old_state, action, state, gs)
        # print(np.all((s_next != goals) == (1 - (s_next == goals))))
        ys = (s_next != goals) * (-1 + self.gamma * np.amax(self.tables[lvl][goals, s_next], axis=-1))
        self.tables[lvl][goals, s, a] += self.lr * (ys - self.tables[lvl][goals, s, a])
    #
    # def update_critic(self, level, s, a, r, s_next, goals, done):
    #     # agent.update_critic(0, old_state, action, -1 * (gs != state), state, gs, 1 * (gs == state))
    #
    #     ys = r + (1 - done) * self.gamma * np.amax(self.tables[level][goals, s_next], axis=-1)
    #     self.tables[level][goals, s, a] += self.lr * (ys - self.tables[level][goals, s, a])
