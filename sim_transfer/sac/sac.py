#Imports
from T1DEKTF import T1DEKTF

import base64
import imageio
import IPython
import matplotlib.pyplot as plt
import os
import reverb
import tempfile

import tensorflow as tf
import numpy as np

from tf_agents.agents.ddpg import critic_rnn_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
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
from tf_agents.trajectories import time_step as ts
from tf_agents.policies import policy_saver
from tf_agents.utils import common

#Folder to save training logs, policies, models, etc. 
tempdir = "st_results/child7/child7_results"

#Reward function for model: Fermi Risk function
def reward(BG_last_hour):
    b = BG_last_hour[-1]
    if 20 <= b < 65:
        return 30-(80-65)*3-(65-b)*10
    elif 65 <= b < 80:
        return 30-(80-b)*3
    elif 80 <= b < 100:
        return 30
    elif 100 <= b < 140:
        return 30-(b-100)*0.2
    elif 140 <= b:
        return 30-(140-100)*0.2-(b-140)*0.5
    else:
        return 0
    #Info: A termination penalty -1e6 is added if the system stops due to hypoglycemia

if __name__ == "__main__":
    ##############   HYPERPARAMETERS   ############### Todo: Make parameters file 
    num_iterations = 1000000 # @param {type:"integer"}

    initial_collect_steps = 1000 # @param {type:"integer"}
    collect_steps_per_iteration = 1 # @param {type:"integer"}
    replay_buffer_capacity = 10000 # @param {type:"integer"}

    batch_size = 256 # @param {type:"integer"}

    critic_learning_rate = 2e-4 # @param {type:"number"}
    actor_learning_rate = 2e-4 # @param {type:"number"}
    alpha_learning_rate = 0 # @param {type:"number"}
    target_update_tau = 1 #0.005 # @param {type:"number"}
    target_update_period = 1 # @param {type:"number"}
    gamma = 0.99 # @param {type:"number"}cla
    reward_scale_factor = 1.0 # @param {type:"number"}

    actor_fc_layer_params = (256,256)
    critic_joint_fc_layer_params = (256,256) 
    lstm = (256,128)

    log_interval = 5000 # @param {type:"integer"}

    num_eval_episodes = 20 # @param {type:"integer"}
    eval_interval = 10000 # @param {type:"integer"}
    vid_interval = 1000 # @param {type:"integer"}

    policy_save_interval = 5000 # @param {type:"integer"}
    deployable_models_dir = "st_results/child7/deploy" # @param {type:"string}
    #################################


    #Load Environment
    collect_env = T1DEKTF(reward_function=reward)
    eval_env = T1DEKTF(reward_function=reward)
    #utils.validate_py_environment(env, episodes=5)

    use_gpu = True
    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)


    #Define Actor, Critic, SAC
    observation_spec, action_spec, time_step_spec = (spec_utils.get_tensor_specs(collect_env))

    with strategy.scope():
        critic_net = critic_rnn_network.CriticRnnNetwork(
        (observation_spec, action_spec), 
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
        observation_spec,
        action_spec,
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
                time_step_spec,
                action_spec,
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


    #Experience replay with reverb (Deepmind)
    #Rate limiter
    #rate_limiter=reverb.rate_limiters.SampleToInsertRatio(samples_per_insert=300.0, min_size_to_sample=3, error_buffer=600.0) #Changed from 3,3 

    table_name = 'Emils_table'
    table = reverb.Table(
        table_name,
        max_size=replay_buffer_capacity,
        sampler=reverb.selectors.Uniform(), 
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(100))

    reverb_server = reverb.Server([table])

    reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
        tf_agent.collect_data_spec,
        sequence_length=2,
        table_name=table_name,
        local_server=reverb_server)


    #Other definitions
    dataset = reverb_replay.as_dataset(
        sample_batch_size=batch_size, num_steps=2).prefetch(50)
    experience_dataset_fn = lambda: dataset

    tf_eval_policy = tf_agent.policy
    eval_policy = py_tf_eager_policy.PyTFEagerPolicy(   
        tf_eval_policy, use_tf_function=True)

    tf_collect_policy = tf_agent.collect_policy
    collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
        tf_collect_policy, use_tf_function=True)

    random_policy = random_py_policy.RandomPyPolicy(
        collect_env.time_step_spec(), collect_env.action_spec())

    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
        reverb_replay.py_client,
        table_name,
        sequence_length=2,
        stride_length=1)

    initial_collect_actor = actor.Actor(
        collect_env,
        random_policy,
        train_step,
        steps_per_run=initial_collect_steps,
        observers=[rb_observer])
    initial_collect_actor.run()

    env_step_metric = py_metrics.EnvironmentSteps()

    collect_actor = actor.Actor(
        collect_env,
        collect_policy,
        train_step,
        steps_per_run=1,
        metrics=actor.collect_metrics(10),
        summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
    observers=[rb_observer, env_step_metric])

    eval_actor = actor.Actor(
        eval_env,
        eval_policy,
        train_step,
        episodes_per_run=num_eval_episodes,
        metrics=actor.eval_metrics(num_eval_episodes),
        summary_dir=os.path.join(tempdir, 'eval'),
    )

    saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)

    # Triggers to save the agent's policy checkpoints.
    learning_triggers = [
        triggers.PolicySavedModelTrigger(
            saved_model_dir,
            tf_agent,
            train_step,
            interval=policy_save_interval),
        triggers.StepPerSecondLogTrigger(train_step, interval=1000),
    ]

    agent_learner = learner.Learner(
        tempdir,
        train_step,
        tf_agent,
        experience_dataset_fn,
    triggers=learning_triggers)

    def get_eval_metrics():
        eval_actor.run()
        results = {}
        for metric in eval_actor.metrics:
            results[metric.name] = metric.result()
        return results

    metrics = get_eval_metrics()

    def log_eval_metrics(step, metrics):
        eval_results = (', ').join(
      '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
        print('step = {0}: {1}'.format(step, eval_results))

    log_eval_metrics(0, metrics)

    ##############
    ########   TRAINING   ######
    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    #  Evaluate the agent's policy once before training.
    avg_return = get_eval_metrics()["AverageReturn"]
    returns = [avg_return]

    for _ in range(num_iterations):
        # Training.
        collect_actor.run()
        loss_info = agent_learner.run(iterations=1)

        # Evaluating.
        step = agent_learner.train_step_numpy

        if eval_interval and step % eval_interval == 0:
            metrics = get_eval_metrics()
            log_eval_metrics(step, metrics)
            returns.append(metrics["AverageReturn"])
            saver = policy_saver.PolicySaver(tf_collect_policy, batch_size=None)
            saver.save(deployable_models_dir + "/" + str(step))

        if log_interval and step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, loss_info.loss.numpy()))

        if vid_interval and step % vid_interval == 0: #Render the eval_env every vid_interval steps
                time_step = eval_env.reset()
                while not time_step.is_last():
                    action_step = eval_actor.policy.action(time_step)
                    #print(action_step.action)
                    time_step = eval_env.step(action_step.action)
                eval_env.render(filename="st_results/child7/child7_renderings/ST" + str(step) + ".png")

    rb_observer.close()
    reverb_server.stop()
