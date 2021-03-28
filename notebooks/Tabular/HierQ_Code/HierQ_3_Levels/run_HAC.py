from collections import deque
import pickle as cpickle

import numpy as np

TEST_FREQ = 2


def get_next_state(state_mat, state, action):
    num_col = state_mat.shape[1]
    num_row = state_mat.shape[0]

    state_row = int(state / num_col)
    state_col = state % num_col

    # If action is "left"
    if action == 0:
        if state_col != 0 and state_mat[state_row][state_col - 1] == 1:
            state -= 1
    # If action is "up"
    elif action == 1:
        if state_row != 0 and state_mat[state_row - 1][state_col] == 1:
            state -= num_col
    # If action is "right"
    elif action == 2:
        if state_col != (num_col - 1) and state_mat[state_row][state_col + 1] == 1:
            state += 1
    # If action is "down"
    else:
        if state_row != (num_row - 1) and state_mat[state_row + 1][state_col] == 1:
            state += num_col

    return state


def run_HAC(state_mat, agent, env, FLAGS, time_limits):
    MAX_LAYER_0_ITR, MAX_LAYER_1_ITR = time_limits[0], time_limits[1]
    max_test_time = state_mat.shape[0] * state_mat.shape[1]
    gs = np.arange(max_test_time)
    test_period = 0

    for episode in range(201):
        if FLAGS.mix and episode % TEST_FREQ == 0:
            FLAGS.test = True

        # Reset environment
        state, solve = 0, False

        # Use below goal for four rooms environment
        goal_row, goal_col = 2, 7
        goal = goal_row * state_mat.shape[1] + goal_col

        if FLAGS.show:
            env.reset_env(state, goal)

        old_subgoal_1 = old_subgoal_2 = -1

        trailing_states = [deque([state], maxlen=MAX_LAYER_0_ITR),
                           deque([state], maxlen=MAX_LAYER_1_ITR * MAX_LAYER_0_ITR)]

        total_steps = t2 = 0
        layer_1_achieved = layer_0_achieved = False
        while not solve:
            initial_state_lay_2 = int(np.copy(state))

            # Track previous subgoal for visualization purposes
            if total_steps > 0:
                old_subgoal_2 = int(np.copy(subgoal_2))

            # Get next subgoal
            subgoal_2 = agent.get_action(initial_state_lay_2, FLAGS.test, 2, goal)
            layer_1_achieved = False

            t1 = 0
            while t1 < MAX_LAYER_1_ITR:
                t1 += 1

                initial_state = int(np.copy(state))
                if total_steps > 0:
                    old_subgoal_1 = int(np.copy(subgoal_1))

                # Get next subgoal
                subgoal_1 = agent.get_action(initial_state, FLAGS.test, 1, subgoal_2)

                # Display subgoal
                if FLAGS.show:
                    env.display_subgoals(subgoal_1, old_subgoal_1, subgoal_2, old_subgoal_2, state, goal)

                t0 = 0
                while t0 < MAX_LAYER_0_ITR: #? and not (layer_0_achieved or layer_1_achieved or solve):
                    t0 += 1

                    old_state = int(np.copy(state))

                    # Get epsilon-greedy action from agent
                    action = agent.get_action(old_state, FLAGS.test, 0, subgoal_1)

                    # Get next state
                    state = get_next_state(state_mat, old_state, action)
                    total_steps += 1

                    if state != old_state:
                        for d in trailing_states:
                            d.append(old_state)

                    # Visualize action if necessary
                    if FLAGS.show:
                        env.step(old_state, state, goal)

                    # Determine reward and whether any of the goals achieved
                    layer_0_achieved = state == subgoal_1
                    layer_1_achieved = state == subgoal_2

                    # print(state, subgoal_1, subgoal_2, goal, action)
                    if state == goal:
                        solve = True
                        print("Episode %d, L2 Itr %d, L1 Itr %d, L0 Itr %d: Goal hit!" % (episode, t2, t1, t0))
                        print("Total Steps: ", total_steps)

                    # Update critic lookup tables
                    if not FLAGS.test:
                        agent.update_critic(0, old_state, action, state, gs)
                        for i in range(1, 3):
                            for s_mem in trailing_states[i - 1]:
                                agent.update_critic(i, s_mem, state, state, gs)
                    if layer_0_achieved or layer_1_achieved or solve:
                        break
                if layer_1_achieved or solve:
                    break

            t2 += 1
            if (solve or total_steps >= max_test_time) and FLAGS.mix and episode % TEST_FREQ == 0:
                FLAGS.test = False
                print("Test Period %d Result: " % test_period, solve)
                test_period += 1
                break

    # Save and Print Q-Table
    cpickle.dump(agent.critic_lay2, open("critic_lay2_table.p", "wb"))
    cpickle.dump(agent.critic_lay1, open("critic_lay1_table.p", "wb"))
    cpickle.dump(agent.critic_lay0, open("critic_lay0_table.p", "wb"))

    print("Critic Tables Saved")
