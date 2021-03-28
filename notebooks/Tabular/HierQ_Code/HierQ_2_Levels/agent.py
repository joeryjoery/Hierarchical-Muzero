from tkinter import *
from tkinter import ttk
import time
import numpy as np
import pickle as cpickle

gamma = 1
step_size = 1
epsilon = 0.25
# epsilon = -1

# Same as np.argmax() except return random index if more than one index with same value

def rand_argmax(arr,goal = None):

    curr_max = arr[0]
    max_ind = [0]

    for i in range(1,len(arr)):
        if arr[i] >= curr_max:
            if arr[i] > curr_max:
                max_ind = [i]
                curr_max = arr[i]
            else:
                max_ind.append(i)

    # Return index from max_ind array
    if goal is None:
        return_ind = np.random.randint(0,len(max_ind))
    else:
        # Check if goal state is in list
        for i in range(len(max_ind)):
            if max_ind[i] == goal:
                return max_ind[i]
        return_ind = np.random.randint(0,len(max_ind))

        
    return max_ind[return_ind]


class Agent():

    def __init__(self,num_squares,num_actions,time_limits,FLAGS):
        if FLAGS.retrain:
            # If retrain, reset critic lookup table for each agent layer
            # self.critic_lay1 = np.zeros((num_squares,num_squares)) 
            self.critic_lay1 = np.ones((num_squares,num_squares)) * -num_squares
            self.critic_lay0 = np.ones((num_squares,num_squares,num_actions)) * -time_limits[0]     
        else:
            # Load critic lookup table
            self.critic_lay1 = cpickle.load(open("critic_lay1_table.p", "rb" ))
            self.critic_lay0 = cpickle.load(open("critic_lay0_table.p", "rb" ))   

            """
            print("HL Q Values in Goal 4, State 8")
            print(self.critic_lay1[8]) 
            print("LL Q Values in Goal 4, State 8")
            print(self.critic_lay0[21][8]) 
            print("LL Q Values in Goal 4, State 3")
            print(self.critic_lay0[21][3]) 
            """  

    def get_action(self,state,FLAGS,layer,goal = None):
        if FLAGS.test or np.random.random_sample() > epsilon:
            # If need action prescribed by policy with no noise
            if layer == 1:
                action = rand_argmax(self.critic_lay1[state],goal)
            elif layer == 0:
                action = rand_argmax(self.critic_lay0[goal][state])
            act_type = "Policy"
        else:
            # Otherwise choose random action 
            if layer == 1:
                action = rand_argmax(self.critic_lay1[state],goal)
                # action = np.random.randint(0,self.critic_lay1.shape[1])
            elif layer == 0:
                action = np.random.randint(0,self.critic_lay0[goal].shape[1])
            act_type = "Random"

        return action, act_type

    # Function updates the goals and rewards in HER transitions
    def process_HER_trans(self,transitions,num_goals):
        
        HER_trans = []  
        # Determine transitions to serve as hindsight goals which will include last state
        indices = np.zeros((num_goals,))
        indices[:num_goals-1] = np.random.randint(0,len(transitions),(num_goals-1,)) 
        indices[num_goals-1] = int(len(transitions)-1)
        # print("Indices: ", indices)
 
        for num in range(num_goals):
            # print("HER Processsing: %d" % num)
            trans_copy = np.copy(transitions)    
            # Determine hindsight goal     
            # print("Index: ", indices[num])   
            goal = trans_copy[int(indices[num])][3]
            for i in range(int(indices[num]) + 1):
                # Update goal value
                trans_copy[i][4] = goal
                # Change rewards and finished boolean if hindsight goal achieved in transition
                if trans_copy[i][3] == goal:
                    trans_copy[i][2] = 0
                    trans_copy[i][5] = True
                HER_trans.append(trans_copy[i])

        return HER_trans

    # def update_critic_lay0(self,old_state,action,reward,next_state,goal,done):
    def update_critic_lay0(self,transition):
        # print("Update Trans: ", transition)
        old_state = transition[0]
        action = transition[1]
        reward = transition[2]
        next_state = transition[3]
        goal = transition[4]
        done = bool(transition[5])
        
       
        # print("Old State: ", old_state)
        # print("Action: ", action)
        # Determine target from Bellman equation
        if not done:
            target = reward + gamma * np.amax(self.critic_lay0[goal][next_state])
            # next_action = np.argmax(self.critic_lay0[goal][next_state])
            # target = reward + gamma * self.critic_lay0[goal][next_state][next_action]
        else:
            target = reward

        # print("Target: ", target)
        # print("Action: ", action)
        # print("Old Q(s,a): ", self.critic_lay0[goal][old_state][action])

        # Push Q(s,a) closer to target 
        self.critic_lay0[goal][old_state][action] = self.critic_lay0[goal][old_state][action] + step_size * (target - self.critic_lay0[goal][old_state][action])

        # print("New Q(s,a): ", self.critic_lay0[goal][old_state][action])


    def update_critic_lay1(self,old_state,action,reward,next_state,goal,done):

        # old_state = int(old_state)
       
        # print("Old State: ", old_state)
        # print("Action: ", action)
        # Determine target from Bellman equation
        if not done:
            target = reward + gamma * np.amax(self.critic_lay1[next_state])
            # next_action = np.argmax(self.critic_lay1[next_state])
            # target = reward + gamma * self.critic_lay1[next_state][next_action]
        else:
            target = reward

        # print("Target1: ", target)
        # print("Action: ", action)
        # print("Old Q1(s,a): ", self.critic_lay1[old_state][action])

        # Push Q(s,a) closer to target 
        self.critic_lay1[old_state][action] = self.critic_lay1[old_state][action] + step_size * (target - self.critic_lay1[old_state][action])

        # print("New Q1(s,a): ", self.critic_lay1[old_state][action])

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



        
        


