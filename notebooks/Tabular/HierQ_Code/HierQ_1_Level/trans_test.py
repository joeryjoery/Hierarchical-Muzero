import numpy as np

trailing_states = []

state = 0
total_steps = 0
MAX_LOW_LEVEL_ITR = 4

for int in range(50):

    for i in range(MAX_LOW_LEVEL_ITR):

        old_state = state
        action = np.random.randint(0,4)
        state = np.random.randint(0,25)
        total_steps += 1

        print("Old State %d, Action %d, Next State %d" % (old_state,action,state))

        trailing_states.append(old_state)
        if len(trailing_states) > MAX_LOW_LEVEL_ITR:
            trailing_states.pop(0)

        if total_steps == 2:
            break

        if i < MAX_LOW_LEVEL_ITR-1 and total_steps > max(i+1,1):

            for j in range(min(total_steps-1,MAX_LOW_LEVEL_ITR-1-i)):
                # for k in range(i+1):
                print([trailing_states[j],state,state])
            
        
        

        
