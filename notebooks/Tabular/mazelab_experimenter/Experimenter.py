import typing
import sys
import time
import datetime
from abc import ABC, abstractmethod

import gym
import tqdm
import numpy as np

from mazelab_experimenter.agents import Agent


def exec_func(func: typing.Callable):
    return func()


class Hook(ABC):
    """ Basic interface for a monitorring Callback that can be used to evaluate an Agent's progress. """
    
    @abstractmethod
    def labels(self) -> typing.Union[str, typing.List]:
        """ Get labels for the aggregated values. """

    @abstractmethod
    def clear(self) -> None:
        """ Clear internal state variables. """
        
    @abstractmethod
    def collect(self, **kwargs) -> None:
        """ Store variables internally """
        
    @abstractmethod
    def aggregate(self, refresh: bool = False, **kwargs) -> typing.Generic:
        """ Aggregate all internal state variables and return the result. """


class PredictionErrorHook(Hook):

    _LABELS = ["Prediction Error"]

    def __init__(self, reference: np.ndarray, get_critic: typing.Callable, f_aggr: typing.Callable = np.mean) -> None:
        """Initialize the monitorring hook with an aggregation function. Defaults to a sample average.

        :param f_aggr: typing.Callable Aggregation function for summarizing collected statistics.
        """
        super().__init__()
        self._reference = reference
        self._get_critic = get_critic
        self._values = list()
        self._f_aggr = f_aggr

    def labels(self) -> typing.List:
        """ Get labels for the aggregated values. """
        return PredictionErrorHook._LABELS

    def clear(self) -> None:
        """ Clear internal state variables. """
        self._values.clear()

    def collect(self, agent: Agent, **kwargs) -> None:
        """ Log whether the agent reached a goal state, its cumulative episode reward, and the episode length. """
        self._values.append(self._get_critic(agent) - self._reference)  # TODO

    def aggregate(self, **kwargs) -> typing.Generic:
        return [self._f_aggr(v) for v in self._values]


class GenericOuterHook(Hook):
    """ Simple episodic monitor for control agents that logs and aggregates the success-rate, cumulative reward, and episode length. """
    
    _LABELS = ["Success Statistic", "Cumulative Reward", "Episode Length"]
    
    def __init__(self, f_aggr: typing.Callable = np.mean) -> None:
        """Initialize the monitorring hook with an aggregation function. Defaults to a sample average.
        
        :param f_aggr: typing.Callable Aggregation function for summarizing collected statistics.
        """
        super().__init__()
        self._success = list()
        self._cumulative = list()
        self._length = list()
        self._f_aggr = f_aggr
        
    def labels(self) -> typing.List:
        """ Get labels for the aggregated values. """
        return GenericOuterHook._LABELS
        
    def clear(self) -> None:
        """ Clear internal state variables. """
        self._success.clear()
        self._cumulative.clear()
        self._length.clear()
        
    def collect(self, success: float, cumulative: float, time: int, **kwargs) -> None:
        """ Log whether the agent reached a goal state, its cumulative episode reward, and the episode length. """
        self._success.append(success)
        self._cumulative.append(cumulative)
        self._length.append(time)
        
    def aggregate(self, **kwargs) -> typing.Tuple[float, float, float]:
        """ Aggregate the logged statistics over the episodes with self._f_aggr in order of self.labels. """
        return [self._f_aggr(v) for v in [self._success, self._cumulative, self._length]]
    
    
def train(_env: gym.Env, _agent: Agent, _num_episodes: int, _agent_kwargs: typing.Dict) -> None:
    """Wrapper to call either the provided agent's own training loop, or to use the generic train implementation.
    
    :param _env: gym.Env Environment to train _agent on.
    :param _agent: Agent A control agent to be trained for _num_episodes.
    :param _num_episodes: int Number of episodes to train for.
    :param _agent_kwargs: dict Misc. arguments that are passed to the _agent's train function.
    """
    if hasattr(_agent, 'train'):
        _agent.train(_env=_env, num_episodes=_num_episodes, **_agent_kwargs)
    else:
        raise NotImplemented("Agent has no implemented training loop.")


def evaluate(_env: gym.Env, _agent: Agent, _num_evals: int, agent_kwargs: typing.Dict, 
             outer_loop_hooks: typing.List[Hook], inner_loop_hooks: typing.Optional[typing.List[Hook]] = None,
             progress_bar: bool = False, clear_outer_hook: bool = True, render: bool = False,
             **kwargs) -> typing.Tuple[typing.List, typing.List]:
    """Defines the core framework for evaluating a control agent on a given environment.
    
    See the benchmark function for extended documentation on Hooks and keyword arguments.
    
    :param _env: gym.Env Environment to evaluate _agent on.
    :param _agent: Agent Control agent that samples actions based on states from _env.
    :param _num_evals: int Number of times to evaluate _agent/ gather statistics. (set to 1 if using greedy-deterministic algorithms!)
    :param agent_kwargs: dict Action sampling arguments for the _agent (e.g., training=False or behaviour_policy=False).
    :param outer_loop_hooks: list of Hooks Monitoring objects that are called episodically.
    :param inner_loop_hooks: list of Hooks Monitoring objects that are called at every step of an episode.
    :param progress_bar: bool Whether to print out a progress bar with tqdm.
    :param clear_outer_hook: bool Whether to refresh the outer hook.
    :param render: bool Whether to visually display the agent's progress (only use for swift debugging!).
    
    :returns: tuple(list, list) The aggregated data of the outer and inner loop Hooks, respectively.
    
    :see: benchmark
    """
    outer_loop_data, inner_loop_data = list(), list()
    
    for i in (tqdm.trange(_num_evals, file=sys.stdout, desc='Evaluation') if progress_bar else range(_num_evals)):
        # Reinitialize environment and monitoring variables after each episode.
        state, g, step, goal_achieved, done = _env.reset(), 0, 0, False, False

        while not done:
            if render:
                _env.render()
                time.sleep(0.1)
                
            # Update to the next state according to the agent's policy.
            a = _agent.sample(state, **agent_kwargs)
            next_state, r, done, _ = _env.step(a)

            # Annotate an episode as done if the agent is actually in a goal-state (not if the time expires).
            if done:
                goal_achieved = _env.unwrapped._is_goal(_env.unwrapped.maze.objects.agent.positions[0])
            
            if inner_loop_hooks:  # If specified yield all inner loop statistics for logging.
                inner_data = dict(
                    env=_env, agent=_agent, state=state, action=a, reward=r, next_state=next_state, 
                    done=done, step=step, cumulative=g, goal_achieved=goal_achieved
                )
                # Collect inner loop data for each hook
                _ = [h.collect(**inner_data) for h in inner_loop_hooks]
                   
            # Update state of control.
            g += r
            step += 1
            state = next_state

        # Store agent episode data.
        if inner_loop_hooks:
            inner_loop_data.append([h.aggregate() for h in inner_loop_hooks])
            for h in inner_loop_hooks:
                h.clear()
            
        outer_data = dict(  # Yield all outer loop statistics for logging.
            agent=_agent, cumulative=g, time=step, success=goal_achieved
        )
        _ = [h.collect(**outer_data) for h in outer_loop_hooks]
        
        # Cleanup environment variables
        _env.close()

    # Process outer loop data before returning.
    outer_loop_data = [h.aggregate(clear=clear_outer_hook) for h in outer_loop_hooks]
    if clear_outer_hook:
        for h in outer_loop_hooks:
            h.clear()
    
    # Return agent evaluation data.
    return outer_loop_data, inner_loop_data


def benchmark(env_id: typing.Union[str, gym.Env], _agent_gen: typing.Callable,
              agent_test_kwargs: typing.Dict, agent_train_kwargs: typing.Dict,
              num_repetitions: int, num_iterations: int, num_episodes: int, num_trials: int,
              evaluation_hooks: typing.List[Hook], verbose: bool = True, **kwargs) -> typing.List:
    """Full functionality for benchmarking a single reinforcement learning.

    The function initializes a fresh gym environment given a string identifier along with a fresh agent, it then repeats a train-test loop
    and logs only evaluation statistics through the evaluation_hooks. The function returns the per test-episode aggregated Hook data. 
    
    The Hook class is a monitoring class similarly to keras.Callbacks that is used here strictly for outer-episode statistics.
    To log inner-loop statistics, use the `evaluate` function instead.
    
    :param env_id: str Environment identifier that has been registered in the OpenAI Gym framework or the env itself.
    :param _agent_gen: typing.Callable A function that yields an Agent object to be trained and evaluated (benchmarked).
    :param agent_test_kwargs: dict Keyword arguments for the agent's action selection during evaluation.
    :param agent_train_kwargs: dict Keyword arguments specific for the agent's training loop.
    :param num_repetitions: int Number of times to repeat the train-test procedure.
    :param num_iterations: int Number of times to train-test the agent.
    :param num_episodes: int Number of times to train the agent for each train-test iteration.
    :param num_trials: int Number of times to evaluate the agent after an iteration of training (set to 1 if greedy-deterministic!).
    :param evaluation_hooks: list of Hook objects that log evaluation statistics of the agent.
    :param verbose: bool Whether to print out a progress bar and ETA for finishing the experiment.

    :returns: list of evaluation data (computed by the evaluation_hooks) for each repetition.
    
    :see: evaluate
    """
    # Initialize agent and environment.
    env = gym.make(env_id) if isinstance(env_id, str) else env_id
    agent = _agent_gen()

    t_0 = time.time()
    repetition_data = list()
    for r in range(num_repetitions):
        # Ensure we have a freshly initialized agent.
        agent.reset()

        if verbose:
            # If specified log time statistics to the console.
            total = time.time() - t_0
            rate = total / r if r else 0
            eta = datetime.timedelta(seconds=int((num_repetitions - r) * rate)) if r else ""
            print(f"-- Benchmarking Repetition {r+1} / {num_repetitions} --- ETA: {str(eta)} "
                  f"--- Rate: {int(rate)} sec/ it --- Total: {total / 60:.2f} min")

        # Test the freshly initialized agent (without parameter updates) and store results.
        data = [
            evaluate(
                _env=env, _agent=agent, _num_evals=num_trials,
                agent_kwargs=agent_test_kwargs,
                outer_loop_hooks=evaluation_hooks,
                clear_outer_hook=True,
                progress_bar=False)[0]
        ]

        for _ in (tqdm.trange(num_iterations, file=sys.stdout, desc="Train-Test loop") if verbose else range(num_iterations)):
            # Train the agent for a number of times
            train(_env=env, _agent=agent, _num_episodes=num_episodes, _agent_kwargs=agent_train_kwargs)
            # Evaluate agent and store data.
            data.append(
                evaluate(
                    _env=env, _agent=agent, _num_evals=num_trials,
                    agent_kwargs=agent_test_kwargs,
                    outer_loop_hooks=evaluation_hooks,
                    clear_outer_hook=True,
                    progress_bar=False)[0]
            )

        # Store training-testing data and reinitialize the trained agent.
        repetition_data.append(data)

    # Return benchmark results.
    return repetition_data
