import numpy as np

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

"""
def rand_argmax(arr):

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
    return_ind = np.random.randint(0,len(max_ind))
    return max_ind[return_ind]
"""
