"""
General utility functions.

- The `find()` function was written by tstanisl on StackOverflow:
  https://stackoverflow.com/questions/7632963/numpy-find-first-index-of-value-fast
"""
import typing
from mazelab.solvers import dijkstra_solver
from enum import Enum

import numpy as np
import gym


class MazeObjects(Enum):
    """ Enumeration object to explicitly provide a common interface for object values among modules. """
    FREE = 0
    OBSTACLE = 1
    GOAL = 2
    AGENT = 3


def compute_optimal_value(_env: gym.Env) -> np.ndarray:
    world = _env.unwrapped.maze.arr.astype(bool)  # 1 = wall, 0 = empty
    m = _env.unwrapped.motions
    g = _env.unwrapped.maze.get_end_pos()[0]

    coords = np.indices(world.shape).reshape(2, -1).T

    solution = np.zeros(len(coords))
    states = np.flatnonzero(~world.ravel())
    for s in states:
        solution[s] = len(dijkstra_solver(impassable_array=world, motions=m, start_idx=coords[s], goal_idx=g))

    return solution


def get_pos(maze: np.ndarray) -> typing.Tuple:
    """ Find the state-space coordinate of the agent from an environment observation. """
    return find(a=maze, predicate=lambda x: x == MazeObjects.AGENT.value)


def rand_argmax(b: np.ndarray, preference: typing.Optional[np.ndarray] = None,
                mask: typing.Optional[np.ndarray] = None, **kwargs) -> int:
    """ a random tie-breaking argmax supplemented with a selection preference and masking functionality. """
    if mask is not None:
        indices = np.flatnonzero(np.isclose(b, b[mask].max()) & mask)
    else:
        indices = np.flatnonzero(np.isclose(b, b.max()))

    if preference is not None:
        pref = np.flatnonzero(indices == preference)
        if len(pref):
            return indices[np.random.choice(pref)]
    return np.random.choice(indices)


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> int:
    """ Compute the Manhattan/ L1 distance between two 1D arrays of uniform dimensions. """
    return np.abs(a - b).sum()


def chebyshev_distance(a: np.ndarray, b: np.ndarray) -> int:
    """ Compute the Chebyshev/ L∞ distance between two 1D arrays of uniform dimensions."""
    return np.max(a - b)


def neumann_neighborhood_size(radius: int) -> int:
    """ Computes the number of tiles in a Von Neumann neighborhood (diamond pattern) given some radius. """
    return radius ** 2 + (radius + 1) ** 2


def moore_neighborhood_size(radius: int) -> int:
    """ Computes the number of tiles in a Moore neighborhood (square pattern) given some radius. """
    return (2 * radius + 1) ** 2


def ravel_moore_index(coords: np.ndarray, radius: int, delta: bool = False) -> np.ndarray:
    """Ravel matrix-coordinates to flattened matrix indices within a Moore Neighborhood.

    Optionally, set delta to true if the given indices are displaced by the Neighborhood center tile.

    :param coords: np.ndarray 2D matrix containing N row-col pairs.
    :param radius: int Maximum Chebyshev distance w.r.t. the Moore Neighborhood center tile.
    :param delta: bool Whether the given coordinates are displaced by the center tile or not.

    :returns: np.ndarray 1D Array of indices of the given coordinates inside the Moore's Neighborhood.
    """
    n = (2 * radius + 1)
    c = coords[:, 0] * n + coords[:, 1]
    return n ** 2 // 2 + c if delta else c


def unravel_moore_index(indices: np.ndarray, radius: int, delta: bool = False) -> np.ndarray:
    """Unravel array-indices to matrix-coordinates within a Moore Neighborhood.

    Optionally, set delta to true if the resulting coordinates should be displaced by the Neighborhood center tile.

    :param indices: np.ndarray 1D array containing ravelled matrix coordinates.
    :param radius: int Maximum Chebyshev distance w.r.t. the Moore Neighborhood center tile.
    :param delta: bool Whether the indices should be displaced by the center tile or not.

    :returns: np.ndarray 2D Array of coordinates for the specified Moore's Neighborhood matrix.
    """
    n = radius * 2 + 1
    coords = np.asarray(np.unravel_index(indices, shape=(n, n)))
    if delta:  # Displace coordinates by center
        coords -= np.asarray(np.unravel_index(n ** 2 // 2, shape=(n, n)))
    return coords


def ravel_neumann_index(coords: np.ndarray, radius: int, delta: bool = False) -> np.ndarray:
    """Ravel delta coordinates to flattened matrix indices within a Von Neumann Neighborhood.

    The coordinates indicate the (row, col) displacement with respect to the center-tile of the Neighborhood.
    This Neighborhood is represented by the Diamond within the (2 * r + 1) * (2 * r + 1) matrix.

    :param coords: np.ndarray 2D matrix containing N row-col pairs.
    :param radius: int Maximum Manhattan distance w.r.t. the Von Neumann Neighborhood center tile.
    :param delta: bool Whether the given coordinates are displaced by the center tile or not.

    :returns: np.ndarray 1D Array of indices of the given coordinates inside the Von Neumann's Neighborhood.
    """
    if not delta:
        coords = coords - radius  # Offset coordinates by the Diamond center-tile.

    center_index = radius * (radius + 1)
    vert_trans = center_index - (radius + 1 - np.abs(coords[:, 0])) * (radius - np.abs(coords[:, 0]))
    return center_index + vert_trans * np.sign(coords[:, 0]) + coords[:, 1]


def unravel_neumann_index(indices: np.ndarray, radius: int, delta: bool = False) -> np.ndarray:
    """Unravel array-indices to matrix-coordinates within a Von Neumann Neighborhood.

    Optionally, set delta to True if the resulting coordinates should be displaced by the Neighborhood center-tile.
    This Neighborhood is represented by the Diamond within the (2 * r + 1) * (2 * r + 1) matrix.

    Warning: does not check whether provided indices exceed the maximum number of tiles.

    :param indices: np.ndarray 1D array containing indices 0 <= i < r^2 + (r + 1)^2 of the Neighborhood.
    :param radius: int Maximum Manhattan distance w.r.t. the Von Neumann Neighborhood center tile.
    :param delta: bool Whether to offset the computed coordinates by the Diamond's center-tile.

    :returns: np.ndarray 2D Matrix of coordinates for the given diamond indices of shape N x (row + col).
    """

    # Maximum number of tiles in a Von Neumann Neighborhood/ diamond.
    m = radius ** 2 + (radius + 1) ** 2

    # Find the nearest root for the indices in the lower diamond, s.t. root < idx < root + 1
    roots = np.floor(np.sqrt(indices))
    # For the squares of the upper diamond, flip the indices and offset the root value by the diamond diameter.
    upper = roots > radius
    roots[upper] = 2 * radius - np.floor(np.sqrt(m - 1 - indices[upper]))

    # Using the row information of roots, compute the center indices of the diamond.
    center = roots * (roots + 1)
    n_offset = 2 * radius - roots[upper]
    center[upper] = (m - 1) - n_offset * (n_offset + 1)  # Correct for decreasing pattern in top diamond.

    # Compute the rows and columns of the diamond's index contained within the 2*r+1 x 2*r+1 matrix.
    rows = roots + (-radius if delta else 0)
    cols = indices - center + (0 if delta else radius)

    return np.stack([rows, cols], axis=1).astype(np.int32)  # Row, Column as matrix columns.


def find(a: np.ndarray, predicate: typing.Callable) -> typing.Optional[typing.Tuple]:
    """
    Get the first index tuple of 'a' that satisfies the given predicate.

    :param a: np.ndarray Some array.
    :param predicate: typing.Callable A function that maps elements of a to {0, 1}.
    :returns: The first tuple representing the index of 'a' that satisfies the predicate or None.
    """
    flat = a.ravel()

    mask = predicate(flat)
    idx = mask.view(bool).argmax() // mask.itemsize

    return np.unravel_index(idx, a.shape, order='C') if flat[idx] else None
