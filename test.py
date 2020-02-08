import argparse

from dqn import DeepQNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game_name", help="name of the game to test, e.g. pong", type=str)
    parser.add_argument("weight_path", help="path from where to load the weights")
    parser.add_argument("--num_frames", default=10000, type=int)
    args = parser.parse_args()

    qnet = DeepQNet(env_name="{}Deterministic-v4".format(args.game_name.title()))
    qnet.load(args.weight_path)
    qnet.play(args.num_frames)
