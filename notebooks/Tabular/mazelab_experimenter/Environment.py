""" Modified from https://github.com/zuoxingdong/mazelab/blob/master/examples/navigation_env.ipynb """
import typing
import inspect
from copy import deepcopy
from collections import deque

import numpy as np

from mazelab import BaseMaze
from mazelab import Object
from mazelab import DeepMindColor as color
from mazelab import VonNeumannMotion
from mazelab import BaseEnv

import mazelab.generators as gen

import gym
from gym.spaces import Box
from gym.spaces import Discrete

from .utils import MazeObjects


class _Maze(BaseMaze):
    """ Wrapper for instantiating a Maze environment. Modified from Zuo Xingdong's original code. """

    def __init__(self, arr: np.ndarray, start_pos: typing.List[typing.Tuple[int, int]],
                 goal_pos: typing.List[typing.Tuple[int, int]]) -> None:
        self.arr = np.copy(arr)
        self._start_idx = np.copy(start_pos)
        self._goal_idx = np.copy(goal_pos)

        # Call super constructor after initializing dependent variables.
        super().__init__()

    def get_start_pos(self) -> typing.List[typing.Tuple[int, int]]:
        return np.copy(self._start_idx)

    def get_end_pos(self) -> typing.List[typing.Tuple[int, int]]:
        return np.copy(self._goal_idx)

    @property
    def size(self) -> typing.Tuple:
        """ Returns the dimensions of the maze. :see: np.ndarray.shape """
        return self.arr.shape

    def make_objects(self) -> typing.Tuple[Object, Object, Object, Object]:
        free = Object(name='free', value=MazeObjects.FREE.value, rgb=color.free, impassable=False,
                      positions=np.stack(np.where(self.arr == MazeObjects.FREE.value), axis=1))
        obstacle = Object(name='obstacle', value=MazeObjects.OBSTACLE.value, rgb=color.obstacle, impassable=True,
                          positions=np.stack(np.where(self.arr == MazeObjects.OBSTACLE.value), axis=1))
        agent = Object(name='agent', value=MazeObjects.AGENT.value, rgb=color.agent, impassable=False, positions=[])
        goal = Object(name='goal', value=MazeObjects.GOAL.value, rgb=color.goal, impassable=False, positions=[])

        return free, obstacle, agent, goal


class _Env(BaseEnv):
    """ Wrapper for instantiating a MazeLab environment as an OpenAI Gym Environment. Modified from Zuo Xingdong's original code. """

    def __init__(self, maze_cls: _Maze, binary_rewards: bool = False, shortest_path_rewards: bool = False,
                 prob: float = 1.0, track_states: bool = False, max_log_memory: int = 1000, **kwargs) -> None:
        super().__init__()

        self.maze = maze_cls
        self.motions = VonNeumannMotion()
        self.reward_probability = prob  # Probability to yield reward upon encountering goal state.
        self.binary_rewards = binary_rewards  # Overrides shortest path.  TODO: Rework dirty reward args.
        self.shortest_path_rewards = shortest_path_rewards

        self.observation_space = Box(low=0, high=len(self.maze.objects), shape=self.maze.size, dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))

        self.steps = 0
        self.track_states = track_states
        if self.track_states:
            self.state_count_memory = deque(maxlen=max_log_memory)
            self.state_count = np.zeros(self.observation_space.shape)

    def step(self, action: int) -> typing.Tuple[np.ndarray, float, bool, dict]:
        motion = self.motions[action]
        coord_t = self.maze.objects.agent.positions[0]
        coord_next = [coord_t[0] + motion[0], coord_t[1] + motion[1]]
        self.steps += 1

        valid = self._is_valid(coord_next)
        if valid:
            self.maze.objects.agent.positions = [coord_next]
        else:
            coord_next = coord_t

        # Reward specification. Reward is either shaped or binary (sparse shortest-path reward).
        done = self._is_goal(coord_next)

        # Reward function cases.
        if self.shortest_path_rewards:
            reward = -1
            if done and np.random.rand() < self.reward_probability:
                reward = 0
        elif self.binary_rewards:
            reward = 0
            if done and np.random.rand() < self.reward_probability:
                reward = 1
        else:
            reward = -1
            if valid:
                reward = -0.01
                if done and np.random.rand() < self.reward_probability:
                    reward = 1

        if self.track_states:
            self.state_count[coord_next[0], coord_next[1]] += 1

        return self.maze.to_value(), reward, done, {'coord': coord_t, 'coord_next': coord_next, 'goal_achieved': done}

    def clone(self):
        return deepcopy(self)

    def reset(self):
        self.steps = 0
        if self.track_states:
            self.state_count_memory.append(np.copy(self.state_count))
            self.state_count[...] = 0

        self.maze.objects.agent.positions, self.maze.objects.goal.positions = self.maze.get_start_pos(), self.maze.get_end_pos()
        return self.maze.to_value()

    def _is_valid(self, position: typing.Tuple[int, int]) -> bool:
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1]
        passable = not self.maze.to_impassable()[position[0]][position[1]]
        return nonnegative and within_edge and passable

    def _is_goal(self, position: typing.Tuple[int, int]) -> bool:
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                return True
        return False

    def get_image(self) -> np.ndarray:
        return self.maze.to_rgb()


class EnvRegister:
    """Static utility class to serve as a wrapper to register custom MazeLab environments in the OpenAI Gym interface.

    """

    _MODULES = [x for x in dir(gen) if not x.startswith('__')]  # All generator module functions as string names.

    @staticmethod
    def get_keyword_args(maze_type: str) -> inspect.Signature:
        """
        Retrieve the environment/ generator specific function arguments for a particular MazeLab environment.
        
        :param maze_type: str One of the module names within EnvRegister._MODULES
        :see: EnvRegister.get_types to retrieve a list of all generators
        :see: mazelab.generators for the backend code of the generators
        """
        mod = [x for x in EnvRegister._MODULES if str(x) == maze_type]
        assert len(mod) == 1, f"Incorrect module specification: {maze_type}"

        return eval(f"inspect.signature(gen.{mod[0]})")

    @staticmethod
    def get_initialization_args() -> str:
        """ 
        Get the initialization arguments needed for instantiating an *agent* in a MazeLab environment.
        :returns: str Example/ dummy dictionary string for formatting the initialization argument for initializing an agent.
        """
        return "{'start_pos': [[i, j]], 'goal_pos': [[i', j'], ...]}"

    @staticmethod
    def get_types() -> typing.List[str]:
        """ Get all available implemented MazeLab environments from `mazelab.generators` """
        return EnvRegister._MODULES

    @staticmethod
    def register(maze_type: str, name: str, env_args: dict, generator_args: dict,
                 initialization_args: dict, time_limit: int = 200, override: bool = False) -> str:
        """ Register an available MazeLab environment with fixed environment parameters as a gym Environment.
        
        This function makes the configured MazeLab environment accessible through `env = gym.make(name)`.
        
        :param maze_type: str One of the available generators given in EnvRegister.
        :param name: str Environment ID to register the MazeLab environment on.
        :param env_args: dict Data structure that contains miscellaneous keyword arguments to the MazeLab Environment.
        :param generator_args: dict Data structure that contains all parameters to call the specified environment generator.
        :param initialization_args: dict Data structure that contains the start and goal states for the agent.
        :param time_limit: int Maximum number of steps before closing the gym Environment.
        :param override: bool If an environment is already registered by 'name', override the existing environment.
        
        :returns: name str Equal to the `name` parameter to allow convenient initialization through `gym.make(EnvRegister.register(...))`
        
        :see: EnvRegister.get_types for a list of all available maze_types.
        :see: EnvRegister.get_keyword_args for retrieving all generator_args needed for each maze_type.
        :see: EnvRegister.get_initialization_args for retrieving all initialization_args for the MazeLab environment.
        """
        if name in gym.envs.registry.env_specs:
            print(
                f"Warning: Environment {name} is already registered in Gym. Override existing environment: {override}")
            if override:
                EnvRegister.unregister(name)
            else:
                return name

        mod = [x for x in EnvRegister._MODULES if x == maze_type]
        assert len(mod) == 1, f"Incorrect module specification: {maze_type}"

        arr = eval(f'gen.{mod[0]}(**{generator_args})')
        maze = _Maze(arr, **initialization_args)

        env_args['maze_cls'] = maze
        gym.register(id=name, entry_point=_Env, max_episode_steps=time_limit, kwargs=env_args)

        return name

    @staticmethod
    def unregister(name: str) -> None:
        """ Unregister a registered environment. 
        Usage of this function can be prevented by registering environments under unambiguous IDs.
        :param name: str Name of the registered environment to remove.
        """
        if name in gym.envs.registry.env_specs:
            del gym.envs.registry.env_specs[name]
