from tkinter import *
from tkinter import ttk
import time
import numpy as np

class Grid_World():

    def __init__(self, state_mat):

        # Create top-level window
        self.root = Tk()
        self.root.title("Grid World")

        # Create canvas to hold grid world
        self.canvas = Canvas(self.root,width = "500",height = "500")
        self.canvas.grid(column=0, row=0, sticky=(N, W, E, S))

        # Create grid world
        self.state_mat = state_mat 

        # Determine pixel length of each block (assume state_mat is square)       
        num_col = state_mat.shape[1]
        pixel_width = 480
        while pixel_width % num_col != 0:
            pixel_width -= 1
        # print("Block Pixel Length: ", pixel_width)

        num_row = state_mat.shape[0]

        block_length = pixel_width / num_col
            
        # Create rectangles    
        for i in range(num_row):
            for j in range(num_col):
                x_1 = 10 + block_length * j
                y_1 = 10 + block_length * i
                x_2 = x_1 + block_length
                y_2 = y_1 + block_length

                if self.state_mat[i][j] == 1:
                    color = "white"
                else:
                    color = "black"
                    
                self.canvas.create_rectangle(x_1,y_1,x_2,y_2,fill=color)
        

    
    def reset_env(self,state,goal):

        # Reset blocks to original colors
        for i in range(self.state_mat.shape[0]):
            for j in range(self.state_mat.shape[1]):

                if self.state_mat[i][j] == 1:
                    color = "white"
                else:
                    color = "black"

                id_num = i * len(self.state_mat[0]) + j + 1
                    
                self.canvas.itemconfig(id_num,fill=color)

        # Change color of agent's current state and goal state
        self.canvas.itemconfig(state+1,fill="blue")
        self.canvas.itemconfig(goal+1,fill="yellow")

        self.root.update()
        time.sleep(0.1)


    def display_subgoals(self,subgoal_1,old_subgoal_1,state,goal):
   
        if subgoal_1 != old_subgoal_1:
            if old_subgoal_1 != -1:
                # If agent currently at old subgoal
                if old_subgoal_1 == state:
                    self.canvas.itemconfig(old_subgoal_1 + 1,fill="blue")
                elif old_subgoal_1 == goal:
                    self.canvas.itemconfig(old_subgoal_1 + 1,fill="yellow")
                else:
                    self.canvas.itemconfig(old_subgoal_1 + 1,fill="white")  
                                       
            
            self.canvas.itemconfig(subgoal_1 + 1,fill="magenta") 
            self.root.update()
        time.sleep(0.1)

            # 

    def step(self,old_state,new_state,goal):

        # If state has changed, update blocks colors
        if new_state != old_state:
            self.canvas.itemconfig(old_state + 1,fill="white") 
        
        if new_state != goal:            
            self.canvas.itemconfig(new_state + 1,fill="blue")
        else:
            self.canvas.itemconfig(new_state + 1,fill="orange")

        self.root.update()

        time.sleep(0.2)

        return



