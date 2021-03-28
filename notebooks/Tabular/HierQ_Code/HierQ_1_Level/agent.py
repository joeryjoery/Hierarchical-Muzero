import numpy as np
import pickle as cpickle

gamma = 1
step_size = 0.8
epsilon = 0.25


def rand_argmax(arr):
    curr_max = arr[0]
    max_ind = [0]

    for i in range(1, len(arr)):
        if arr[i] >= curr_max:
            if arr[i] > curr_max:
                max_ind = [i]
                curr_max = arr[i]
            else:
                max_ind.append(i)

    # Return index from max_ind array
    return_ind = np.random.randint(0, len(max_ind))
    return max_ind[return_ind]


class Agent:

    def __init__(self, num_squares, num_actions, FLAGS):
        if FLAGS.retrain:
            # If retrain, reset critic lookup table
            self.critic = np.ones((num_squares, num_actions)) * -num_squares
        else:
            # Load critic lookup table
            self.critic = cpickle.load(open("critic_table.p", "rb"))

    def get_action(self, state, FLAGS):
        if FLAGS.test or np.random.random_sample() > epsilon:
            # If testing, use action with largest Q-value
            # action = np.argmax(self.critic[state])
            action = rand_argmax(self.critic[state])
        else:
            # Choose random action
            action = np.random.randint(0, self.critic.shape[1])

        return action

    def update_critic(self, old_state, action, reward, next_state, goal, done):

        target = reward + (1 - int(done)) * gamma * np.amax(self.critic[next_state])

        # Push Q(s,a) closer to target 
        self.critic[old_state][action] += step_size * (target - self.critic[old_state][action])

    def get_policy(self, state_mat):

        num_states = self.critic.shape[0]
        num_cols = int(np.sqrt(num_states))

        policy = []
        for i in range(num_cols):
            policy.append(["" for x in range(num_cols)])

        for i in range(num_states):
            opt_action = np.argmax(self.critic[i])
            direction_list = ["left", "up", "right", "down"]
            action = direction_list[opt_action]
            # print("Policy action: ", action)

            row_num = int(i / num_cols)
            col_num = i % num_cols

            policy[row_num][col_num] = action

        # Change action to "N/A" for all wall squares in grid world        
        for row in range(state_mat.shape[0]):
            for col in range(state_mat.shape[1]):
                if state_mat[row][col] == 0:
                    policy[row][col] = "N/A"

        return policy
