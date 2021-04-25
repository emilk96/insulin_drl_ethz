import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, LSTM, Dropout
from tensorflow.keras.models import Sequential

import gym
import argparse
import numpy as np
import copy

import tqdm
import collections
import matplotlib.pyplot as plt
from gym.envs.registration import register

tf.keras.backend.set_floatx('float64')

## Reference values
# gamma = 0.99
# update_interval = 5
# actor_lr = 0.0005
# critic_lr = 0.0001

class Actor:
    def __init__(self, state_dim, action_dim, action_bound, std_bound, actor_lr):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = std_bound
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(actor_lr)

    def create_model(self):
      initializer = tf.keras.initializers.GlorotNormal()
      state_input = Input((self.state_dim,1))
      lstm1 = LSTM(4, return_sequences=True)(state_input)
      dropout1 = Dropout(0.1)(lstm1)
      lstm2 = LSTM(2, return_sequences=False)(dropout1)
      dense1 = Dense(32, activation='relu')(lstm2)
      out_mu = Dense(self.action_dim, activation='tanh', kernel_initializer=initializer)(dense1)  
      mu_output = Lambda(lambda x: x * self.action_bound)(out_mu)
      std_output = Dense(self.action_dim, activation='softplus', kernel_initializer=initializer)(dense1)
      return tf.keras.models.Model(state_input, [mu_output, std_output])

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim, 1])
        mu, std = self.model.predict(state)
        mu, std = mu[0], std[0]
        return np.random.normal(mu, std, size=self.action_dim)

    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / \
            var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def compute_loss(self, mu, std, actions, advantages):
        log_policy_pdf = self.log_pdf(mu, std, actions)
        loss_policy = log_policy_pdf * advantages
        return tf.reduce_sum(-loss_policy)

    def train(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            mu, std = self.model(states, training=True)
            loss = self.compute_loss(mu, std, actions, advantages)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Critic:
    def __init__(self, state_dim, critic_lr):
        self.state_dim = state_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(critic_lr)

    def create_model(self):
        #initializer = tf.keras.initializers.GlorotNormal()
      model = Sequential()
      model.add(LSTM(16, return_sequences=True))
      model.add(Dropout(0.1))
      model.add(LSTM(8, return_sequences=False))
      model.add(Dropout(0.1))
      model.add(Dense(1, activation='linear'))
      return model

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Agent:
    def __init__(self, env, gamma=0.99, update_interval=120, actor_lr=0.001, critic_lr=0.0005):
        self.env = env
        self.gamma = gamma
        self.update_interval = update_interval
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.state_dim = 30 #hardcoded -> self.env.observation_space[0] not used because it is set to 1
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        self.std_bound = [1e-2, 1.0]
        
        self.actor = Actor(self.state_dim, self.action_dim,
                           self.action_bound, self.std_bound, self.actor_lr)
        self.critic = Critic(self.state_dim, self.critic_lr)

    def td_target(self, reward, next_state, done):
        if done:
            return reward
        v_value = self.critic.model.predict(
            np.reshape(next_state, [1, self.state_dim, 1]))
        return np.reshape(reward + self.gamma * v_value[0], [1, 1])

    def adv(self, td_targets, baselines):  
        return td_targets - baselines 

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def train(self, max_episodes=100000):
        for ep in range(max_episodes):
            print("Training...")
            state_batch = []
            action_batch = []
            td_target_batch = []
            adv_batch = []
            episode_reward, done = 0, False

            #Get state and normalize 
            self.env.seed()
            state = self.env.reset()
            state = state[0] / 350.0
            _state = np.full((1,self.state_dim,1), 0)
            _state = np.delete(_state, 0, axis=1)
            _state = np.append(_state, [[[state]]], axis=1)
            state = np.reshape(_state, [1, self.state_dim, 1])

            while not done:
                action_unclipped = self.actor.get_action(state)
                action = np.clip(action_unclipped, 0, self.action_bound) 
                added_state, reward, done, _ = self.env.step(action)

                added_state = added_state[0]

                #Normalization of state and reward 
                added_state = added_state / 350.0
                reward = (reward + 1.0) / 2.0

                action = np.reshape(action, [1, self.action_dim])
                next_state = copy.deepcopy(state)
                next_state = np.delete(next_state, 0, axis=1)
                next_state = np.append(next_state, [[[added_state]]], axis=1)
                next_state = np.reshape(next_state, [1, self.state_dim, 1])
                reward = np.reshape(reward, [1, 1])

                td_target = self.td_target(reward, next_state, done) #(reward+8)/8
                advantage = self.adv(td_target, self.critic.model.predict(state))

                state_batch.append(state)
                action_batch.append(action)
                td_target_batch.append(td_target)
                adv_batch.append(advantage)

                if len(state_batch) >= self.update_interval or done:
                    states = self.list_to_batch(state_batch)
                    actions = self.list_to_batch(action_batch)
                    td_targets = self.list_to_batch(td_target_batch)
                    advantages = self.list_to_batch(adv_batch)

                    actor_loss = self.actor.train(states, actions, advantages)
                    critic_loss = self.critic.train(states, td_targets)
                    #print ("CL={}   AL={}".format(critic_loss, actor_loss))

                    state_batch = []
                    action_batch = []
                    td_target_batch = []
                    adv_batch = []

                episode_reward += reward[0][0]
                state = 0
                state = next_state

            print('EP{} EpisodeReward={}'.format(ep, episode_reward))
            if ep % 1 == 0:
                self.env.render(filename="renderings/EP" + str(ep) + ".png")
            else: 
                pass