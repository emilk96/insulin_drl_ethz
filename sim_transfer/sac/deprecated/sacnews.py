# pip3 install tf-agents[reverb]
# pip3 install pybullet
#Necessary to cite tf-agents, tf, reverb in thesis 

from T1DEKTF import T1DEKTF

import base64
import imageio
import IPython
import matplotlib.pyplot as plt
import os
import tempfile
import PIL.Image

import tensorflow as tf

from tf_agents.agents.ddpg import critic_rnn_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments import suite_pybullet
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.policies import policy_saver
from tf_agents.utils import common
from tf_agents.trajectories import trajectory

from tf_agents import specs
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.replay_buffers import py_uniform_replay_buffer
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step
from tf_agents.policies import random_tf_policy

import numpy as np

tempdir = "sac_results"

def reward(BG_last_hour):
    b = BG_last_hour[-1]
    c0 = 3.35506
    c1 = 0.8353
    c2 = 3.7932
    risk = 10 * (c0 * (np.log(b)**c1 - c2))**2
    return -risk  

    #Info: A termination penalty -1e6 is added if the system stops due to hypoglycemia

if __name__ == "__main__":
    ##############   HYPERPARAMETERS   ############### Todo: Make parameters file 
    num_iterations = 1000000 # @param {type:"integer"}

    initial_collect_steps = 10000 # @param {type:"integer"}
    collect_steps_per_iteration = 1 # @param {type:"integer"}
    replay_buffer_capacity = 100000 # @param {type:"integer"}

    batch_size = 256 # @param {type:"integer"}

    critic_learning_rate = 3e-4 # @param {type:"number"}
    actor_learning_rate = 3e-4 # @param {type:"number"}
    alpha_learning_rate = 3e-4 # @param {type:"number"}
    target_update_tau = 1 #0.005 # @param {type:"number"}
    target_update_period = 1 # @param {type:"number"}
    gamma = 0.99 # @param {type:"number"}cla
    reward_scale_factor = 1.0 # @param {type:"number"}

    actor_fc_layer_params = (256,256) #(128,)
    critic_joint_fc_layer_params = (256,256) #(128,)
    lstm = (256,128)

    log_interval = 5000 # @param {type:"integer"}

    num_eval_episodes = 20 # @param {type:"integer"}
    eval_interval = 1 # @param {type:"integer"}
    vid_interval = 1 # @param {type:"integer"}

    policy_save_interval = 5000 # @param {type:"integer"}
    deployable_models_dir = "deploy" # @param {type:"string}
    #################################


    #Load Environment
    collect_py_env = T1DEKTF(reward_function=reward)
    eval_py_env = T1DEKTF(reward_function=reward)
    collect_env = tf_py_environment.TFPyEnvironment(collect_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    #utils.validate_py_environment(env, episodes=5)

    use_gpu = True
    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)

    #Define Actor, Critic, SAC    
    with strategy.scope():
        critic_net = critic_rnn_network.CriticRnnNetwork(
        (collect_env.time_step_spec().observation, collect_env.action_spec()), 
        observation_conv_layer_params=None,
        observation_fc_layer_params=None, 
        action_fc_layer_params=None,
        joint_fc_layer_params=critic_joint_fc_layer_params, 
        lstm_size=lstm, 
        output_fc_layer_params=None,
        activation_fn=tf.keras.activations.relu, 
        kernel_initializer='glorot_uniform',
        last_kernel_initializer='glorot_uniform', 
        rnn_construction_fn=None,
        rnn_construction_kwargs=None, 
        name='CriticRnnNetwork')

    with strategy.scope():
        actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork( 
        collect_env.time_step_spec().observation,
        collect_env.action_spec(),
        preprocessing_layers=None,
        preprocessing_combiner=None, 
        conv_layer_params=None, 
        input_fc_layer_params=None, 
        input_dropout_layer_params=None, 
        lstm_size=lstm,
        output_fc_layer_params=actor_fc_layer_params, 
        activation_fn=tf.keras.activations.relu,
        dtype=tf.float32, 
        discrete_projection_net=None,
        continuous_projection_net=(tanh_normal_projection_network.TanhNormalProjectionNetwork), 
        rnn_construction_fn=None,
        rnn_construction_kwargs=None, 
        name='ActorDistributionRnnNetwork')

    with strategy.scope():
        train_step = train_utils.create_train_step()

        tf_agent = sac_agent.SacAgent(
                collect_env.time_step_spec(),
                collect_env.action_spec(),
                actor_network=actor_net,
                critic_network=critic_net,
                actor_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=actor_learning_rate),
                critic_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=critic_learning_rate),
                alpha_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=alpha_learning_rate),
                target_update_tau=target_update_tau,
                target_update_period=target_update_period,
                td_errors_loss_fn=common.element_wise_huber_loss,
                gamma=gamma,
                reward_scale_factor=reward_scale_factor,
                train_step_counter=train_step)

        tf_agent.initialize()

    #Initialize replay buffer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    tf_agent.collect_data_spec,
    batch_size=collect_env.batch_size,
    max_length=1000) #350000

    
    #Name policies
    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy

    random_tf_policy = random_tf_policy.RandomTFPolicy(
        collect_env.time_step_spec(), collect_env.action_spec())

    saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)

    replay_observer = [replay_buffer.add_batch]

    collect_steps_per_iteration = 2
    collect_op = dynamic_step_driver.DynamicStepDriver(
        collect_env,
        tf_agent.collect_policy,
        observers=replay_observer,
        num_steps=collect_steps_per_iteration).run()

    dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2)

    iterator = iter(dataset)

    #Optimize by wrapping some of the code in a graph using TF function.
    tf_agent.train = common.function(tf_agent.train)

    # Reset the train step
    tf_agent.train_step_counter.assign(0)
    def compute_avg_return(environment, policy, num_episodes=10):
        total_return = 0.0
        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]


    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
    returns = [avg_return]

    for _ in range(num_iterations):
        print(_)

        # Collect a few steps using collect_policy and save to the replay buffer.
        #collect_data(collect_env, tf_agent.collect_policy, replay_buffer, collect_steps_per_iteration)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = tf_agent.train(experience).loss

        step = tf_agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)

        if step % vid_interval == 0: #Render the eval_env every vid_interval steps
                time_step = eval_env.reset()
                while not time_step.is_last():
                    action_step = tf_agent.policy.action(time_step)
                    #print(action_step.action)
                    time_step = eval_env.step(action_step.action)
                eval_env.render(filename="sac_renderings/ST" + str(step) + ".png")





