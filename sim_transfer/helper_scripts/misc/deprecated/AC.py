#!/usr/bin/env python
# coding: utf-8

# In[2]:




# In[5]:


import collections
import gym
import numpy as np
import tensorflow as tf
import tqdm

from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple


# Create the environment
env = gym.make("CartPole-v0")

# Set seed for experiment reproducibility
seed = 42
env.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()


# In[6]:


class ActorCritic(tf.keras.Model):
  """Combined actor-critic network."""

  def __init__(
      self, 
      num_actions: int, 
      num_hidden_units: int):
    """Initialize."""
    super().__init__()

    self.common = layers.Dense(num_hidden_units, activation="relu")
    self.actor = layers.Dense(num_actions)
    self.critic = layers.Dense(1)

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = self.common(inputs)
    return self.actor(x), self.critic(x)


# In[7]:


num_actions = env.action_space.n  # 2
num_hidden_units = 128

model = ActorCritic(num_actions, num_hidden_units)


# In[8]:


# Wrap OpenAI Gym's `env.step` call as an operation in a TensorFlow function.
# This would allow it to be included in a callable TensorFlow graph.

def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns state, reward and done flag given an action."""

  state, reward, done, _ = env.step(action)
  return (state.astype(np.float32), 
          np.array(reward, np.int32), 
          np.array(done, np.int32))


def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
  return tf.numpy_function(env_step, [action], 
                           [tf.float32, tf.int32, tf.int32])


# In[9]:


def run_episode(
    initial_state: tf.Tensor,  
    model: tf.keras.Model, 
    max_steps: int) -> List[tf.Tensor]:
  """Runs a single episode to collect training data."""

  action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

  initial_state_shape = initial_state.shape
  state = initial_state

  for t in tf.range(max_steps):
    # Convert state into a batched tensor (batch size = 1)
    state = tf.expand_dims(state, 0)

    # Run the model and to get action probabilities and critic value
    action_logits_t, value = model(state)

    # Sample next action from the action probability distribution
    action = tf.random.categorical(action_logits_t, 1)[0, 0]
    action_probs_t = tf.nn.softmax(action_logits_t)

    # Store critic values
    values = values.write(t, tf.squeeze(value))

    # Store log probability of the action chosen
    action_probs = action_probs.write(t, action_probs_t[0, action])

    # Apply action to the environment to get next state and reward
    state, reward, done = tf_env_step(action)
    state.set_shape(initial_state_shape)

    # Store reward
    rewards = rewards.write(t, reward)

    if tf.cast(done, tf.bool):
      break

  action_probs = action_probs.stack()
  values = values.stack()
  rewards = rewards.stack()

  return action_probs, values, rewards


# In[10]:


def get_expected_return(
    rewards: tf.Tensor, 
    gamma: float, 
    standardize: bool = True) -> tf.Tensor:
  """Compute expected returns per timestep."""

  n = tf.shape(rewards)[0]
  returns = tf.TensorArray(dtype=tf.float32, size=n)

  # Start from the end of `rewards` and accumulate reward sums
  # into the `returns` array
  rewards = tf.cast(rewards[::-1], dtype=tf.float32)
  discounted_sum = tf.constant(0.0)
  discounted_sum_shape = discounted_sum.shape
  for i in tf.range(n):
    reward = rewards[i]
    discounted_sum = reward + gamma * discounted_sum
    discounted_sum.set_shape(discounted_sum_shape)
    returns = returns.write(i, discounted_sum)
  returns = returns.stack()[::-1]

  if standardize:
    returns = ((returns - tf.math.reduce_mean(returns)) / 
               (tf.math.reduce_std(returns) + eps))

  return returns


# In[11]:


huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

def compute_loss(
    action_probs: tf.Tensor,  
    values: tf.Tensor,  
    returns: tf.Tensor) -> tf.Tensor:
  """Computes the combined actor-critic loss."""

  advantage = returns - values

  action_log_probs = tf.math.log(action_probs)
  actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

  critic_loss = huber_loss(values, returns)

  return actor_loss + critic_loss


# In[12]:


optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


@tf.function
def train_step(
    initial_state: tf.Tensor, 
    model: tf.keras.Model, 
    optimizer: tf.keras.optimizers.Optimizer, 
    gamma: float, 
    max_steps_per_episode: int) -> tf.Tensor:
  """Runs a model training step."""

  with tf.GradientTape() as tape:

    # Run the model for one episode to collect training data
    action_probs, values, rewards = run_episode(
        initial_state, model, max_steps_per_episode) 

    # Calculate expected returns
    returns = get_expected_return(rewards, gamma)

    # Convert training data to appropriate TF tensor shapes
    action_probs, values, returns = [
        tf.expand_dims(x, 1) for x in [action_probs, values, returns]] 

    # Calculating loss values to update our network
    loss = compute_loss(action_probs, values, returns)

  # Compute the gradients from the loss
  grads = tape.gradient(loss, model.trainable_variables)

  # Apply the gradients to the model's parameters
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  episode_reward = tf.math.reduce_sum(rewards)

  return episode_reward


# In[13]:


# get_ipython().run_cell_magic('time', '', "\nmax_episodes = 10000\nmax_steps_per_episode = 1000\n\n# Cartpole-v0 is considered solved if average reward is >= 195 over 100 \n# consecutive trials\nreward_threshold = 195\nrunning_reward = 0\n\n# Discount factor for future rewards\ngamma = 0.99\n\nwith tqdm.trange(max_episodes) as t:\n  for i in t:\n    initial_state = tf.constant(env.reset(), dtype=tf.float32)\n    episode_reward = int(train_step(\n        initial_state, model, optimizer, gamma, max_steps_per_episode))\n\n    running_reward = episode_reward*0.01 + running_reward*.99\n\n    t.set_description(f'Episode {i}')\n    t.set_postfix(\n        episode_reward=episode_reward, running_reward=running_reward)\n\n    # Show average episode reward every 10 episodes\n    if i % 10 == 0:\n      pass # print(f'Episode {i}: average reward: {avg_reward}')\n\n    if running_reward > reward_threshold:  \n        break\n\nprint(f'\\nSolved at episode {i}: average reward: {running_reward:.2f}!')")


# # In[16]:


# # Render an episode and save as a GIF file

# from IPython import display as ipythondisplay
# from PIL import Image
# from pyvirtualdisplay import Display


# display = Display(visible=0, size=(400, 300))
# display.start()


# def render_episode(env: gym.Env, model: tf.keras.Model, max_steps: int): 
#   screen = env.render(mode='rgb_array')
#   im = Image.fromarray(screen)

#   images = [im]

#   state = tf.constant(env.reset(), dtype=tf.float32)
#   for i in range(1, max_steps + 1):
#     state = tf.expand_dims(state, 0)
#     action_probs, _ = model(state)
#     action = np.argmax(np.squeeze(action_probs))

#     state, _, done, _ = env.step(action)
#     state = tf.constant(state, dtype=tf.float32)

#     # Render screen every 10 steps
#     if i % 10 == 0:
#       screen = env.render(mode='rgb_array')
#       images.append(Image.fromarray(screen))

#     if done:
#       break

#   return images


# # Save GIF image
# images = render_episode(env, model, max_steps_per_episode)
# image_file = 'cartpole-v0.gif'
# # loop=0: loop forever, duration=1: play each frame for 1ms
# images[0].save(
#     image_file, save_all=True, append_images=images[1:], loop=0, duration=1)


# # In[ ]:


# import tensorflow_docs.vis.embed as embed
# embed.embed_file(image_file)


# In[ ]:





# In[ ]:





# In[ ]:




