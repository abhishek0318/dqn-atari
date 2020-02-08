import argparse

from dqn import DeepQNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game_name", help="name of the game to train on, e.g. pong", type=str)
    parser.add_argument("weight_save_path", help="path to where save the game", type=str)
    parser.add_argument("--training_frames", default=10000000, type=int)
    parser.add_argument("--minibatch_size", default=32, type=int)
    parser.add_argument("--replay_memory_size", default=1000000, type=int)
    parser.add_argument("--target_network_update_frequency", default=10000, type=int)
    parser.add_argument("--discount_factor", default=0.99, type=float)
    parser.add_argument("--learning_rate", default=0.0001, type=float)
    parser.add_argument("--initial_exploration", default=1, type=float)
    parser.add_argument("--final_exploration", default=0.1, type=float)
    parser.add_argument("--final_exploration_frame", default=1000000, type=int)
    parser.add_argument("--replay_start_size", default=50000, type=int)
    parser.add_argument("--checkpoint", action="store_true")

    args = parser.parse_args()
    checkpoint_path = args.weight_save_path if args.checkpoint else None

    qnet = DeepQNet(env_name="{}Deterministic-v4".format(args.game_name.title()))
    qnet.train(training_frames=args.training_frames,
               minibatch_size=args.minibatch_size,
               replay_memory_size=args.replay_memory_size,
               target_network_update_frequency=args.target_network_update_frequency,
               discount_factor=args.discount_factor,
               learning_rate=args.learning_rate,
               initial_exploration=args.initial_exploration,
               final_exploration=args.final_exploration,
               final_exploration_frame=args.final_exploration_frame,
               replay_start_size=args.replay_start_size,
               checkpoint_path=checkpoint_path)
    qnet.save(args.weight_save_path)
