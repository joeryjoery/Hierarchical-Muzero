import matplotlib.pyplot as plt
import numpy as np
import gym

from mazelab_experimenter.agents import TabularQLearning


def plotQPolicy(env: gym.Env, agent: TabularQLearning) -> None:

    # Q-table visualization
    fig = plt.figure(figsize=(8, 8))
    img = env.unwrapped.get_image()

    # Get the trained agent's policy over the entire table for visualizing its greedy policy.
    actions = np.argmax(agent._q_table, axis=-1)

    # Get undefined coordinates for removing clutter in the image.
    mask = env.unwrapped.maze.objects.obstacle.positions.tolist() + env.unwrapped.maze.objects.goal.positions.tolist()
    mask = set(map(tuple, mask))

    positions, motions = list(), list()
    for i, a in enumerate(actions.ravel()):
        pos = np.unravel_index(i, shape=actions.shape)
        positions.append(pos)
        motions.append(env.unwrapped.motions[a] if pos not in mask else [0, 0])

    # Define a grid and extract for each grid-point a correct Quiver/ Arrow
    # direction for visualizing the action-selection policy.
    X, Y = list(zip(*positions))
    V, U = list(zip(*motions))

    U = np.asarray(U)
    V = np.negative(V)  # Motion is inverted vertically.

    # Show the maze image with overlayed quivers. The minlength=0.0 argument removes all quivers on invalid states.
    plt.imshow(img)
    plt.quiver(Y, X, U, V, scale=20, minlength=0.0, pivot='middle', linewidth=0.1)

    return fig


def plotQTable(env: gym.Env, agent: TabularQLearning, v_min: float = 0.0, v_max: float = 1.0) -> None:

    # Q-table visualization
    fig = plt.figure(figsize=(8, 8))

    # Get the trained agent's policy over the entire table for visualizing its greedy policy.
    values = np.max(agent._q_table, axis=-1)

    # Get goal/ terminal position and set value to max reward.
    goal = env.unwrapped.maze.objects.goal.positions
    values[goal[:, 0], goal[:, 1]] = v_max

    # Show the maze image with overlayed quivers. The minlength=0.0 argument removes all quivers on invalid states.
    plt.imshow(values, cmap=plt.get_cmap('inferno'), vmin=v_min, vmax=v_max)

    return fig
