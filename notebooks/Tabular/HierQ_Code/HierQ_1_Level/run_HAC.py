from tkinter import *
from tkinter import ttk
import time
import numpy as np
from grid_world import Grid_World
from setup_flags import set_up
import pickle as cpickle

TEST_FREQ = 2
MAX_LOW_LEVEL_ITR = 50


def get_next_state(state_mat,state, action):

        num_col = state_mat.shape[1]
        num_row = state_mat.shape[0]

        state_row = int(state/num_col)    
        state_col = state % num_col
        

        # print("State row: ", state_row)
        # print("State col: ", state_col)

        # If action is "left"
        if action == 0:
            if state_col != 0 and state_mat[state_row][state_col - 1] == 1:
                # print("Moving Left")
                state -= 1
        # If action is "up"
        elif action == 1:
            if state_row != 0 and state_mat[state_row - 1][state_col] == 1:
                # print("Moving Up")
                state -= num_col
        # If action is "right"
        elif action == 2:
            if state_col != (num_col - 1) and state_mat[state_row][state_col + 1] == 1:
                # print("Moving Right")
                state += 1
        # If action is "down"
        else:
            if state_row != (num_row - 1) and state_mat[state_row + 1][state_col] == 1:
                # print("Moving Down")
                state += num_col

        return state



def run_HAC(state_mat,agent,env,FLAGS):

    test_period = 0

    max_test_time = state_mat.shape[0] * state_mat.shape[1]

    for episode in range(101):

        if FLAGS.mix and episode % TEST_FREQ == 0:
            # print("Episode % ", episode)
            FLAGS.test = True

        """
        if FLAGS.test:
            print("Episode %d Critic Table: " % episode)
            print(agent.critic)
        """

        # Reset environment
        state = 0
        # goal = state_mat.shape[0] * state_mat.shape[1] - 1

        
        goal_row = 2
        goal_col = 7
        goal = goal_row * state_mat.shape[1] + goal_col
        

        if FLAGS.show:
            env.reset_env(state,goal)

        # time.sleep(10)

        done = False

        total_steps = 0

        t = 0

        while not done:

            old_state = np.copy(state)
            old_state = int(old_state)

            # Get epsilon-greedy action from agent
            action = agent.get_action(state,FLAGS)
            # action = np.random.randint(0,4)
            
            act_strings = ["left","up","right","down"]
            # print("Step %d: " % t, act_strings[action])

            # print("New Q(s,a) v1: ", agent.critic[old_state][action])

            # Get next state
            # print("Episode: %d Testing: " % episode, FLAGS.test)

            state = get_next_state(state_mat,np.copy(old_state),action)
            # print("Next State: ", state)

            total_steps += 1


            # Visualize action if necessary
            if FLAGS.show:
                env.step(old_state,state,goal)
            
            # Determine reward and whether goal achieved
            reward = -1   
            if state == goal:
                reward = 0
                done = True
                print("Episode %d, Step %d: Goal hit!" % (episode,t))
                print("Total Steps: ", total_steps)

            # Update critic lookup table
            if not FLAGS.test:
                agent.update_critic(old_state,action,reward,state,goal,done)
                # print("New Q(s,a) v3: ", agent.critic[old_state][action])

            t += 1
            
            if FLAGS.mix and episode % TEST_FREQ == 0 and (done or total_steps >= max_test_time):          
                FLAGS.test = False
                print("Test Episode %d Result: " % test_period, done)
                test_period += 1
                break

    # Save and Print Q-Table
    cpickle.dump(agent.critic,open("critic_table.p","wb"))
    print("Critic Table Saved")

    # print("Q-Table: ")
    # print(agent.critic)

    """
    print("\nPolicy: ")
    policy = agent.get_policy(state_mat)
    for i in policy:
        print(i)
    print("\n")
    """
            

        

        
