from tkinter import *
from tkinter import ttk
import time
import numpy as np
from grid_world import Grid_World
from setup_flags import set_up
from run_HAC import run_HAC
from agent import Agent

# Determine mode of agent (train, test, mix, visualization on/off)
FLAGS = set_up()

# Design grid world ("0" = wall, "1" = open space)
len_array = 11
state_mat = np.ones((len_array,len_array))
# world_len = 8
# state_mat = np.ones((world_len,world_len))

# Uncomment lines below to create four rooms environment
state_mat[:,5] = 0
state_mat[5,:] = 0
state_mat[5,2:4] = 1
state_mat[7:9,5] = 1
state_mat[5,7:9] = 1


# print(state_mat)

# Create grid world visualization (if in --show mode)
if FLAGS.show:
    env = Grid_World(state_mat)
else:
    env = None

# Create agent's Q-function lookup table.  Dimensions of table will be (# of blocks x # actions).  Rows of table that correspond to blocks containing walls will thus not be updated.
num_blocks = state_mat.shape[0] * state_mat.shape[1]
num_actions = 4
time_limits = [4]
agent = Agent(num_blocks, num_actions,time_limits, FLAGS)

run_HAC(state_mat,agent,env,FLAGS,time_limits)

# env.root.mainloop()

