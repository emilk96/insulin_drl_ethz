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


class Actor:
    def __init__(self, state_dim, feature_dim, action_dim, action_bound, std_bound, actor_lr, clip_ratio):
        self.state_dim = state_dim
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = std_bound
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(actor_lr)
        self.clip_ratio = clip_ratio

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim, self.feature_dim])
        mu, std = self.model.predict(state)
        action = np.random.normal(mu[0], std[0], size=self.action_dim)
        action = np.clip(action, 0, self.action_bound)
        log_policy = self.log_pdf(mu, std, action)

        return log_policy, action

    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / \
            var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def create_model(self):
        state_input = Input((self.state_dim,self.feature_dim))
        lstm1 = LSTM(8, return_sequences=True)(state_input)
        lstm2 = LSTM(4, return_sequences=False)(lstm1)
        dense1 = Dense(32, activation='relu')(lstm2)
        out_mu = Dense(self.action_dim, activation='tanh')(dense1)  
        mu_output = Lambda(lambda x: x * self.action_bound)(out_mu)
        std_output = Dense(self.action_dim, activation='softplus')(dense1)
        return tf.keras.models.Model(state_input, [mu_output, std_output])

    def compute_loss(self, log_old_policy, log_new_policy, actions, gaes):
        ratio = tf.exp(log_new_policy - tf.stop_gradient(log_old_policy))
        gaes = tf.stop_gradient(gaes)
        clipped_ratio = tf.clip_by_value(
            ratio, 1.0-self.clip_ratio, 1.0+self.clip_ratio)
        surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)
        return tf.reduce_mean(surrogate)

    def train(self, log_old_policy, states, actions, gaes):
        with tf.GradientTape() as tape:
            mu, std = self.model(states, training=True)
            log_new_policy = self.log_pdf(mu, std, actions)
            loss = self.compute_loss(
                log_old_policy, log_new_policy, actions, gaes)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Critic:
    def __init__(self, state_dim, feature_dim, critic_lr):
        self.state_dim = state_dim
        self.feature_dim = feature_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(critic_lr)

    def create_model(self):
        model = Sequential()
        model.add(LSTM(8, return_sequences=True))
        model.add(LSTM(4, return_sequences=False))
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
    def __init__(self, env, \
             gamma=0.999, \
             update_interval=3, \
             actor_lr=0.001, \
             critic_lr=0.0005, \
             clip_ratio = 0.1, \
             lmbda = 0.95, \
             epochs = 3 \
             ):

        self.env = env
        self.state_dim = 16
        self.feature_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        self.std_bound = [1e-2, 1.0]
        self.gamma = gamma
        self.lmbda = lmbda
        self.update_interval = update_interval
        self.epochs = epochs
        
        self.actor_opt = tf.keras.optimizers.Adam(actor_lr)
        self.critic_opt = tf.keras.optimizers.Adam(critic_lr)
        self.actor = Actor(self.state_dim, self.feature_dim, self.action_dim,
                           self.action_bound, self.std_bound, actor_lr, clip_ratio)
        self.critic = Critic(self.state_dim, self.feature_dim, critic_lr)

    def gae_target(self, rewards, v_values, next_v_value, done):
        n_step_targets = np.zeros_like(rewards)
        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        forward_val = 0

        if not done:
            forward_val = next_v_value

        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + self.gamma * forward_val - v_values[k]
            gae_cumulative = self.gamma * self.lmbda * gae_cumulative + delta
            gae[k] = gae_cumulative
            forward_val = v_values[k]
            n_step_targets[k] = gae[k] + v_values[k]
        return gae, n_step_targets

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def train(self, max_episodes=1000):
        for ep in range(max_episodes):
            print("Training...")
            state_batch = []
            action_batch = []
            reward_batch = []
            old_policy_batch = []

            episode_reward, done = 0, False

            self.env.seed()
            _state = self.env.reset()
            _state = _state[0] / 350.0
            state = np.full((1,self.state_dim,self.feature_dim), 0)
            for i in range (self.state_dim):
                state = np.delete(state, [0,0], axis=1)
                state = np.append(state, [[[_state, 0.0]]], axis=1) #no meal at reset
                state = np.reshape(state, [1, self.state_dim, self.feature_dim])
            
            while not done:
                log_old_policy, action = self.actor.get_action(state)

                update_cgm, reward, done, info = self.env.step(action)
                update_meal = info['meal']
            
                update_cgm = update_cgm[0]
                update_cgm = update_cgm / 350.0
                update_meal = update_meal / 200.0
                reward = (reward + 1.0) / 2.0

                state = np.reshape(state, [1, self.state_dim, self.feature_dim])
                action = np.reshape(action, [1, 1])

                next_state = copy.deepcopy(state)
                next_state = np.delete(next_state, [0,0], axis=1)
                next_state = np.append(next_state, [[[update_cgm, update_meal]]], axis=1)
                next_state = np.reshape(next_state, [1, self.state_dim, self.feature_dim])

                reward = np.reshape(reward, [1, 1])
                log_old_policy = np.reshape(log_old_policy, [1, 1])

                state_batch.append(state)
                action_batch.append(action)
                reward_batch.append(reward)
                old_policy_batch.append(log_old_policy)

                if len(state_batch) >= self.update_interval or done:
                    states = self.list_to_batch(state_batch)
                    actions = self.list_to_batch(action_batch)
                    rewards = self.list_to_batch(reward_batch)
                    old_policys = self.list_to_batch(old_policy_batch)

                    v_values = self.critic.model.predict(states)
                    next_v_value = self.critic.model.predict(next_state)

                    gaes, td_targets = self.gae_target(
                        rewards, v_values, next_v_value, done)

                    for epoch in range(self.epochs):
                        actor_loss = self.actor.train(
                            old_policys, states, actions, gaes)
                        critic_loss = self.critic.train(states, td_targets)

                    state_batch = []
                    action_batch = []
                    reward_batch = []
                    old_policy_batch = []

                episode_reward += reward[0][0]
                state = 0
                state = next_state

            print('EP{} EpisodeReward={}'.format(ep, episode_reward))
            if ep % 1 == 0:
                self.env.render(filename="pporenderings/EP" + str(ep) + ".png")
            else: 
                pass 
