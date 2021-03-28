import argparse

# Function takes input from user to determine mode of agent.  Options include whether agent will be in training (noise added to policy), testing (no noise added), or a mix.  User can also indicate whether episode should be visualized.

def set_up():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--show',
        default=False,
        action='store_true',
        help='At what point to render the cart environment'
    )


    parser.add_argument(
        '--retrain',
        default=False,
        action='store_true',
        help='Whether to start training from scratch again or not'
    )

    parser.add_argument(
        '--test',
        default=False,
        action='store_true',
        help='Test more or no (true = no training updates)'
    )

    parser.add_argument(
        '--mix',
        default=False,
        action='store_true',
        help='Mix in testing episodes'
    )

    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS
