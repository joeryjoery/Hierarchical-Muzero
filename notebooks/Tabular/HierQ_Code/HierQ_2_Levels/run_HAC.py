from tkinter import *
from tkinter import ttk
import time
import numpy as np
from grid_world import Grid_World
from setup_flags import set_up
import pickle as cpickle


TEST_FREQ = 2


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



def run_HAC(state_mat,agent,env,FLAGS,time_limits):

    MAX_LOW_LEVEL_ITR = time_limits[0]

    if FLAGS.test:  
        print("\nSubgoal Q Values: ")
        print(agent.critic_lay1)
        print("\n")  
        # print("Subgoal Q Values State 0: ", agent.critic_lay1[0])
        # print("Subgoal Q Values State 2: ", agent.critic_lay1[2])

    test_period = 0

    max_test_time = state_mat.shape[0] * state_mat.shape[1]

    for episode in range(201):

        if FLAGS.mix and episode % TEST_FREQ == 0:
            FLAGS.test = True
            """
            print("\n\nSubgoal Q Table: ")
            print(agent.critic_lay1)
            print("\n\n")
            """
        # Reset environment
        state = 0

        # Use below goal state for regular grid world
        # goal = state_mat.shape[0] * state_mat.shape[1] - 1
        
        # Use below goal state for four rooms grid world
        goal_row = 2
        goal_col = 7
        goal = goal_row * state_mat.shape[1] + goal_col
        

        if FLAGS.show:
            env.reset_env(state,goal)

        high_level_achieved = False

        old_subgoal_1 = -1

        t1 = 0

        trailing_states = []

        total_steps = 0

        while not high_level_achieved:

            initial_state = np.copy(state)
            initial_state = int(initial_state)
            
            if t1 > 0:
                old_subgoal_1 = int(np.copy(subgoal_1))

            # Get next subgoal
            subgoal_1, act_1_type = agent.get_action(initial_state,FLAGS,1,goal)

            # Display subgoal
            if FLAGS.show:
                env.display_subgoals(subgoal_1,old_subgoal_1,state,goal)

            # print("Next Subgoal: ", subgoal_1)
            # time.sleep(2)

            # Print Subgoal
            num_cols = state_mat.shape[1]
            row_num = int(subgoal_1 / num_cols)
            col_num = subgoal_1 % num_cols
            # print("\n HL Itr %d Subgoal: %d or (%d,%d)" % (t1,subgoal_1, row_num,col_num))

            low_level_achieved = False

            # Below list serves as temporary storage of trans to be processed during HER            
            trans_lay0 = []


            for t0 in range(MAX_LOW_LEVEL_ITR):

                old_state = np.copy(state)
                old_state = int(old_state)

                # Get epsilon-greedy action from agent
                action, act_0_type = agent.get_action(old_state,FLAGS,0,subgoal_1)
                # action = np.random.randint(0,4)
                
                act_strings = ["left","up","right","down"]
                """
                if act_0_type is not "Random":
                    print("HL Itr %d, LL Itr %d Policy Action: %d" % (t1,t0,action), act_strings[action])
                else:
                    print("HL Itr %d, LL Itr %d Random Action: %d" % (t1,t0,action), act_strings[action])
                """
                

                # Get next state
                state = get_next_state(state_mat,old_state,action)
                # print("Next State: ", state)
                # print("\n\nEpisode %d, HL Itr %d, LL Itr %d" % (episode, t1, t0))
                # print("Old State: %d, Action %d, New State: %d, Subgoal: %d" % (old_state,action,state,subgoal_1))
                
                total_steps += 1

                if state != old_state:
                    trailing_states.append(old_state)
                if len(trailing_states) > (MAX_LOW_LEVEL_ITR):
                    trailing_states.pop(0)

                # Visualize action if necessary
                if FLAGS.show:
                    env.step(old_state,state,goal)
                
                # Determine reward and whether goal achieved
                reward_0 = -1   
                if state == subgoal_1:
                    reward_0 = 0
                    low_level_achieved = True
                    # print("Subgoal hit!")

                reward_1 = -1
                if state == goal:
                    reward_1 = 0
                    high_level_achieved = True
                    print("Episode %d, HL Itr %d, LL Itr %d: Goal hit!" % (episode,t1,t0))
                    print("Total Steps: ", total_steps)

                # Update critic lookup table
                if not FLAGS.test:

                    # Create layer 0 transitions
                    
                    # Create series of transitions evaluating performance of action given every possible subgoal state
                    
                    num_states = state_mat.shape[0] * state_mat.shape[1]

                    for i in range(num_states):
                        if i != state:
                            hindsight_trans_lay0 = [old_state,action,-1,state,i,False] 
                        else:
                            hindsight_trans_lay0 = [old_state,action,0,state,i,True]
                        agent.update_critic_lay0(np.copy(hindsight_trans_lay0)) 
                    
                    # print("State %d Q-Values: " % old_state, agent.critic_lay0[subgoal_1][old_state])
  
                    
                    # Create layer 1 transitions

                    # Create high level hindsight transitions in which each of the past MAX_LOW_LEVEL_ITR states serves as the inital state and "state" serves as the subgoal                  
                    for i in range(len(trailing_states)):
                        agent.update_critic_lay1(trailing_states[i],state,reward_1,state,goal,high_level_achieved)
                    
                    # Print latest Q-values
                    # print("State %d, Subgoal %d Q Values: " % (old_state,state), agent.critic_lay0[state][old_state])
                
                if low_level_achieved or high_level_achieved:
                    break           
                

            # Increment high level iterator
            t1 += 1

            if (high_level_achieved or total_steps >= max_test_time) and FLAGS.mix and episode % TEST_FREQ == 0:

                FLAGS.test = False
                print("Test Period %d Result: " % test_period, high_level_achieved)
                test_period += 1
                    
                break
                

    # Save and Print Q-Table
    cpickle.dump(agent.critic_lay1,open("critic_lay1_table.p","wb"))
    cpickle.dump(agent.critic_lay0,open("critic_lay0_table.p","wb"))
    
    print("Critic Tables Saved")

    """
    print("Q-Table: ")
    print(agent.critic)

    print("\nPolicy: ")
    policy = agent.get_policy(state_mat)
    for i in policy:
        print(i)
    print("\n")
    """
            

        

        
