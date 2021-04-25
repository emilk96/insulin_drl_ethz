import tensorflow as tf
from T1DEKTF import T1DEKTF
from tf_agents.environments import tf_py_environment
import mpu.io
from statistics import mean 
import numpy as np


def inference(policy, env, render_env):
  num_episodes = 10
  log = []
  l_means = []
  h_means = []
  for ep in range(num_episodes):
    ep_log = []
    time_step = env.reset()
    policy_state = policy.get_initial_state(batch_size=1)
    while not time_step.is_last():
        last_cgm = np.array(time_step.observation)[0,119]#change to last
        ep_log.append(last_cgm)
        policy_step = policy.action(time_step, policy_state)
        policy_state = policy_step.state
        time_step = env.step(policy_step.action)
    normal_bg, low_bg, high_bg = render_env.render(filename="test"+str(ep)+ ".png")

    #Logging
    l_means.append(low_bg)
    h_means.append(high_bg)
    log.append(ep_log)

  l_mean = mean(l_means)
  h_mean = mean(h_means)
  print("Hypo: ", l_mean*100, ", Hyper: ", h_mean*100, ", TIR: ", (1-l_mean-h_mean)*100)
  return log
    
if __name__ == "__main__":
    render_env = T1DEKTF()
    env = tf_py_environment.TFPyEnvironment(render_env)

    policy_dir = "st_results/adolescent1/deploy/110000"
    policy = tf.compat.v2.saved_model.load(policy_dir)

    log = inference(policy, env, render_env)
    log = np.array(log)
    log = log.astype('float64')
    final_log = log.tolist()
    mpu.io.write('adol1.json', final_log)

