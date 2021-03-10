"""
Implements the abstract Coach class for defining the data sampling procedures for MuZero neural network training.

Notes:
 - Base implementation done.
 - Documentation 15/11/2020
"""
import typing
from datetime import datetime

import numpy as np
import tensorflow as tf

from Coach import CoachBase
from Agents import HierarchicalMuZeroPlayer
from HMuZero.NetworkHierarchy import TwoLevelNetworkHierarchy
from HMuZero.PolicyHierarchy import PolicyHierarchy
from utils import DotDict
from utils.game_utils import GameState
from utils.selfplay_utils import sample_batch
from utils.selfplay_utils import ParameterScheduler
from utils.HierarchicalUtils import GoalState, HierarchicalGameHistory, get_goal_samples, get_muzero_goal_samples, \
    get_action_samples, get_muzero_action_samples, goal_achieved
from utils import debugging


class HierarchicalCoach(CoachBase):
    """
    Implement base Coach class to define proper data-batch sampling procedures and logging objects.
    """

    def __init__(self, game, neural_net: TwoLevelNetworkHierarchy, args: DotDict,
                 run_name: typing.Optional[str] = None) -> None:
        """
        Initialize the class for self-play. This inherited method initializes tensorboard logging and defines
        helper variables for data batch sampling.

        The super class is initialized with the proper search engine and agent-interface. (MuZeroMCTS, MuZeroPlayer)

        :param game: Game Implementation of Game class for environment logic.
        :param neural_net: HierarchicalMuZero Implementation of HierarchicalMuZero class for inference.
        :param args: DotDict Data structure containing parameters for self-play.
        :param run_name: str Optionally provide a run-name for the TensorBoard log-files. Default is current datetime.
        """
        super().__init__(game=game, neural_net=neural_net, args=args,
                         search_engine=PolicyHierarchy, player=HierarchicalMuZeroPlayer)

        # Initialize tensorboard logging.
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

        self.log_dir = f"out/logs/HierarchicalMuZero/" + run_name
        self.file_writer = tf.summary.create_file_writer(self.log_dir + "/metrics")
        self.file_writer.set_as_default()

        # Define helper variables.
        self.return_forward_observations = (neural_net.net_args.dynamics_penalty > 0 or args.latent_decoder)
        self.observation_stack_length = neural_net.net_args.observation_length

        # TODO: Create exploration/ parameter schedules for agents.
        # Initialize MCTS visit count exponentiation factor schedule.
        self.temp_schedule = ParameterScheduler(self.args.temperature_schedule)
        self.update_temperature = self.temp_schedule.build()

        self.temp_by_weights = self.temp_schedule.args.by_weight_update

    def update_parameters(self, training_data: typing.List[HierarchicalGameHistory]) -> None:
        """ Add functionality for logging latest episode's goal statistics before backpropagation. """
        self.neural_net.monitor.log_goal_statistics(training_data[-self.args.num_episodes:])

        # backpropagation
        super().update_parameters(training_data)

    def sampleBatch(self, histories: typing.List[HierarchicalGameHistory]) -> typing.List:
        sample_coordinates, sample_weight = sample_batch(histories, n=self.neural_net.net_args.batch_size,
                                                         prioritize=False)

        f_goal = get_muzero_goal_samples if self.args.plan_goals else get_goal_samples
        f_action = get_muzero_action_samples if self.args.plan_actions else get_action_samples

        goal_samples = f_goal(self, histories, sample_coordinates, sample_weight)
        action_samples = f_action(self, histories, sample_coordinates, sample_weight)

        return [goal_samples, action_samples]

    def update_goal(self, current_state: GameState, goal: GoalState) -> GoalState:
        if goal.goal is None:
            return goal

        goal.age += 1

        norm = lambda x: x - self.game.obs_low / (self.game.obs_high - self.game.obs_low)
        goal.achieved = goal_achieved(
            norm(current_state.observation.ravel()), norm(goal.goal.ravel()), self.args.goal_error)

        return goal

    def executeEpisode(self) -> HierarchicalGameHistory:
        """
        Perform one episode of self-play for gathering data to train neural networks on.

        The implementation details of the neural networks/ agents, temperature schedule, data storage
        is kept highly transparent on this side of the algorithm. Hence for implementation details
        see the specific implementations of the function calls.

        At every step we record a snapshot of the state into a GameHistory object, this includes the observation,
        MCTS search statistics, performed action, and observed rewards. After the end of the episode, we close the
        GameHistory object and compute internal target values.

        :return: GameHistory Data structure containing all observed states and statistics required for network training.
        """
        history = HierarchicalGameHistory()
        state = self.game.getInitialState()
        step = 0

        # Dummy goal to be updated at the first episode step.
        goal, goal_inference_statistics = GoalState.empty(None), None

        while not state.done and step < self.args.max_episode_moves:
            if debugging.RENDER:  # Display visualization of the environment if specified.
                self.game.render(state)

            # Update MCTS visit count temperature according to an episode or weight update schedule.
            temp_g = self.update_temperature(self.neural_net.goal_net.steps if self.temp_by_weights else step)
            temp_a = self.update_temperature(self.neural_net.action_net.steps if self.temp_by_weights else step)

            # Inference as a joint policy.
            goal = self.update_goal(state, goal)
            if goal.achieved or goal.age >= self.args.goal_horizon:
                goal.end_state = state.observation
                goal, goal_inference_statistics = self.search_engine.sample_goal(state, history, temp_g)
            state.action, action_inference_statistics = self.search_engine.sample_action(state, history, goal, temp_a)

            # Environment transition and store observations.
            next_state, r = self.game.getNextState(state, state.action)
            history.capture(state, next_state, r, goal, goal_inference_statistics, action_inference_statistics)

            # Update state of control
            state = next_state
            step += 1
            goal_inference_statistics = None  # Clear goal inference statistics for memory efficiency.

        # Cleanup environment and GameHistory
        self.game.close(state)
        history.terminate()
        history.compute_returns(self.args.gamma, self.args.n_steps)

        return history
