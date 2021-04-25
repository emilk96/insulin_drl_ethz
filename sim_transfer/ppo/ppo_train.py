import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda

import gym
from gym.envs.registration import register

import argparse
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import collections

from ppo_model import Actor, Critic, Agent
import T1DEK_envs

def reward(BG_last_hour):
    # Attention, reward is changend by minutes alive in train script
    if BG_last_hour[-1] < 80 and BG_last_hour[-1] >=70:
        return -0.8+(BG_last_hour[-1]-80.0)/100.0
    elif BG_last_hour[-1] < 100 and BG_last_hour[-1] >=80:
        return 0
    elif BG_last_hour[-1] < 140 and BG_last_hour[-1] >=100:
        return 1
    elif BG_last_hour[-1] < 180 and BG_last_hour[-1] >=140:
        return 0
    elif BG_last_hour[-1] <= 300 and BG_last_hour[-1] >=180:
        return -0.4-(BG_last_hour[-1]-180.0)/200.0
    else:
        return -1


if __name__ == "__main__":
    #Setup environment
    register(
	id='T1DEK-v0',
	entry_point='T1DEK_envs.envs:T1DEK',
	kwargs={'patient_name': 'adolescent#002', 'reward_function': reward})

    env = gym.make('T1DEK-v0')

    #Begin training
    agent = Agent(env)
    agent.train()
    env.close()
