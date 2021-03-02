import typing
from datetime import datetime

import tensorflow as tf

from Coach import CoachBase
from Agents import ModelFreePlayer
from HMuZero.NetworkHierarchy import TwoLevelNetworkHierarchy
from HMuZero.PolicyHierarchy import PolicyHierarchy
from utils import DotDict
from utils.selfplay_utils import sample_batch, MFGameHistory
from utils.selfplay_utils import ParameterScheduler
from utils import debugging


class MFCoach(CoachBase):

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
                         search_engine=PolicyHierarchy, player=ModelFreePlayer)

        # Initialize tensorboard logging.
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

        self.log_dir = f"out/logs/ModelFree/" + run_name
        self.file_writer = tf.summary.create_file_writer(self.log_dir + "/metrics")
        self.file_writer.set_as_default()

        # TODO: Create exploration/ parameter schedules for DDPG/ DQN agents.
        # Initialize MCTS visit count exponentiation factor schedule.
        self.explore_schedule = ParameterScheduler(self.args.exploration_schedule)
        self.update_schedule = self.explore_schedule.build()

    def sampleBatch(self, histories: typing.List[MFGameHistory]) -> typing.List:
        sample_coordinates, sample_weight = sample_batch(histories, n=self.neural_net.net_args.batch_size,
                                                         prioritize=False)

        examples = [(
            histories[h_i].observations[i],              # state
            self.neural_net.goal_net.current_goal.goal,  # goal
            histories[h_i].actions[i],                   # actions
            histories[h_i].rewards[i],                   # rewards
            histories[h_i].next_observations[i],         # next state
            self.neural_net.goal_net.current_goal.goal,  # next goal
            int(i+1 == len(histories[h_i])),             # done
        )
            for h_i, i in sample_coordinates
        ]

        empty_examples = [()] * len(sample_coordinates)

        # For UVFA interface, [goal examples, action examples]
        return [empty_examples, examples]

    def executeEpisode(self) -> MFGameHistory:
        history = MFGameHistory()
        state = self.game.getInitialState()
        step = 0

        while not state.done and step < self.args.max_episode_moves:
            if debugging.RENDER:  # Display visualization of the environment if specified.
                self.game.render(state)

            # Update exploration schedule.
            epsilon = self.update_schedule(self.neural_net.action_net.steps)

            state.action, action_inference_statistics = self.search_engine.sample_action(
                state, history, self.neural_net.goal_net.current_goal, epsilon)

            # Environment transition and store observations.
            next_state, r = self.game.getNextState(state, state.action)
            history.capture(state, next_state, r)

            # Update state of control
            state = next_state
            step += 1

        # Cleanup environment and GameHistory
        self.game.close(state)
        history.terminate()

        return history





