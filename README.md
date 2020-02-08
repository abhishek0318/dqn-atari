# Playing Atari games using Deep Q Network (PyTorch)

This repository contains PyTorch implementation of the Deep Q Network paper published by DeepMind. Though many modifications have been proposed to DQNs since then, I have tried to stick to the original description.

![Trained agent playing breakout](./breakout_test.gif?raw=true "Trained agent playing breakout")

## Papers

If you are looking to start with Deep Q Learning, I would recomment you go through these papers in this order. These papers do not assume much prior knowledge except for basics of Reinforcement Learning and Deep Learning.

1. [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)  
2. [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)  

## Installation

Install the requirements using the following command. The code is tested on Python 3.6+. 

> pip install -r requirements.txt

## Overview

[`dqn.py`](./dqn.py) contains the majority of the code including the agent's network and the training algorithm (`train` function). 
[`train.py`](./train.py) and [`test.py`](./test.py) are the scripts for training and testing the agent respectively.

### Training the agent

If you are training on a server with sufficient RAM, just run `python train.py <game_name> <weight_save_path>`. The default parameters will give you good results. The list of `game_name` values can be found [here](https://gym.openai.com/envs/#atari).

If you want to train the agent on your laptop, you will have to reduce the `replay memory size` and `replay start size` to prevent out of memory error.

> python train.py pong pong.pt --replay_memory_size 10000 --replay_start_size 2000 --checkpoint

Training the agent on GTX 1080 takes about 12 hours.

```
$ python train.py --help
usage: train.py [-h] [--training_frames TRAINING_FRAMES]
                [--minibatch_size MINIBATCH_SIZE]
                [--replay_memory_size REPLAY_MEMORY_SIZE]
                [--target_network_update_frequency TARGET_NETWORK_UPDATE_FREQUENCY]
                [--discount_factor DISCOUNT_FACTOR]
                [--learning_rate LEARNING_RATE]
                [--initial_exploration INITIAL_EXPLORATION]
                [--final_exploration FINAL_EXPLORATION]
                [--final_exploration_frame FINAL_EXPLORATION_FRAME]
                [--replay_start_size REPLAY_START_SIZE] [--checkpoint]
                game_name weight_save_path

positional arguments:
  game_name             name of the game to train on, e.g. pong
  weight_save_path      path to where save the game

optional arguments:
  -h, --help            show this help message and exit
  --training_frames TRAINING_FRAMES
  --minibatch_size MINIBATCH_SIZE
  --replay_memory_size REPLAY_MEMORY_SIZE
  --target_network_update_frequency TARGET_NETWORK_UPDATE_FREQUENCY
  --discount_factor DISCOUNT_FACTOR
  --learning_rate LEARNING_RATE
  --initial_exploration INITIAL_EXPLORATION
  --final_exploration FINAL_EXPLORATION
  --final_exploration_frame FINAL_EXPLORATION_FRAME
  --replay_start_size REPLAY_START_SIZE
  --checkpoint
```

### Testing the agent

To see your trained agent playing game, just run `python test.py <game_name> <weight_path>`. Note that the weight should correspond to the same game only.

```
$ python test.py --help
usage: test.py [-h] [--num_frames NUM_FRAMES] game_name weight_path

positional arguments:
  game_name             name of the game to test, e.g. pong
  weight_path           path from where to load the weights

optional arguments:
  -h, --help            show this help message and exit
  --num_frames NUM_FRAMES
```

### Trained Models

This repository contains trained weight for `breakout` game. To watch the agent play `breakout`, simply run,

> python test.py breakout breakout.pt

These weights were trained using this command,

> python train.py breakout breakout.pt --replay_memory_size 50000 --replay_start_size 10000


### Learning curves

Following are the learning curves for the `breakout` game. The graph on the left is the plot of each minibatch loss vs the frame number. There is a clear pattern of spike in loss after every 40k frames. These correspond to update in the target q function.

The graph on the right is the plot of reward in a episode (episode_reward) vs the frame number. The average episode reward stagnates after a while, though the agent continues to learn. This is in line with what is described in the paper. 

![Breakout learning curves](./breakout_learning_graph.png?raw=true "Breakout learning curves")
