import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda

import gym
from gym.envs.registration import register

import argparse
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import collections

from A2C_rnn import Actor, Critic, Agent
import T1DEK_envs


# def reward(BG_last_hour):
#     if BG_last_hour[-1] < 80 and BG_last_hour[-1] >=70:
#         return -0.8+(BG_last_hour[-1]-80.0)/100.0
#     elif BG_last_hour[-1] < 100 and BG_last_hour[-1] >=80:
#         return 0
#     elif BG_last_hour[-1] < 140 and BG_last_hour[-1] >=100:
#         return 1
#     elif BG_last_hour[-1] < 180 and BG_last_hour[-1] >=140:
#         return 0
#     elif BG_last_hour[-1] <= 300 and BG_last_hour[-1] >=180:
#         return -0.4-(BG_last_hour[-1]-180.0)/200.0
#     else:
#         return -1

def reward(BG_last_hour):
    b = BG_last_hour[-1]
    c0 = 3.35506
    c1 = 0.8353
    c2 = 3.7932
    risk = 10 * (c0 * (np.log(b)**c1 - c2))**2
    return -risk  


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



### Different reward functions
# def bg_reward(BG_last_hour):
#     if BG_last_hour[-1] > 350:
#         return -50
#     elif BG_last_hour[-1] > 170 and BG_last_hour[-1] <= 350:
#         return 0
#     elif BG_last_hour[-1] < 90 and BG_last_hour[-1] >= 70:
#         return -2
#     elif BG_last_hour[-1] < 70:
#         return -50
#     else:
#         return 3

# def d_reward(BG_last_hour):
#     if len(BG_last_hour) < 2:
#         return 0
    
#     else:
#         x = BG_last_hour[-2:]
#         y = np.arange(0,len(x))
#         x = np.array(x)

#         dy=np.diff(y,1)
#         dx=np.diff(x,1)
#         yfirst=dy/dx
#         # xfirst=0.5*(x[:-1]+x[1:])

#         # dyfirst=np.diff(yfirst,1)
#         # dxfirst=np.diff(xfirst,1)
#         # ysecond=dyfirst/dxfirst

#         #ysecond_max = np.max(ysecond)
#         yfirst_abs = np.absolute(yfirst)
#         yfirst_abs = np.clip(yfirst_abs, 0, 2) 
#         return -yfirst_abs[0]

# # Reward function for the model 
# def reward(BG_last_hour):
#     bgr = bg_reward(BG_last_hour)
#     dr = d_reward(BG_last_hour)
#     print(dr)
#     return (bgr+dr)