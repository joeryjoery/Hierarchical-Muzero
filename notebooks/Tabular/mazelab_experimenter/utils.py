"""
General utility functions.

- The `find()` function was written by tstanisl on StackOverflow:
  https://stackoverflow.com/questions/7632963/numpy-find-first-index-of-value-fast
"""
import typing
from enum import Enum

import numpy as np


class MazeObjects(Enum):
    """ Enumeration object to explicitly provide a common interface for object values among modules. """
    FREE = 0
    OBSTACLE = 1
    GOAL = 2
    AGENT = 3


def rand_argmax(b: np.ndarray, preference: typing.Optional[np.ndarray] = None, **kwargs) -> int:
    """ a random tie-breaking argmax"""
    indices = np.flatnonzero(np.isclose(b, b.max()))

    pref = np.flatnonzero(indices == preference)
    if len(pref):
        return indices[np.random.choice(pref)]
    return np.random.choice(indices)


def find(a: np.ndarray, predicate: typing.Callable):
    """
    Parameters
    ----------
    a : numpy.ndarray
        Input data, can be of any dimension.

    predicate : function
        An element-wise function which results in a boolean np.ndarray.

    Returns
    -------
    index_generator : int
        A generator of (indices, data value) tuples which make the predicate
        True.

    See Also
    --------
    where, nonzero

    Notes
    -----
    This function is best used for finding the first, or first few, data values
    which match the predicate.

    Examples
    --------
    >>> a = np.sin(np.linspace(0, np.pi, 200))
    >>> result = find(a, predicate=lambda x: x > 0.9)
    >>> result
    71
    >>> np.where(a > 0.9)[0][0]
    71
    """
    flat = a.ravel()

    mask = predicate(flat)
    idx = mask.view(bool).argmax() // mask.itemsize

    return np.unravel_index(idx, a.shape, order='C') if flat[idx] else -1
