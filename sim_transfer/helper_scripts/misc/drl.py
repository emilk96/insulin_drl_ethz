import gym
from gym.envs.registration import register
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def custom_reward(BG_last_hour):
    if BG_last_hour[-1] > 180:
        return -1
    elif BG_last_hour[-1] < 70:
        return -2
    else:
        return 1


register(
    id='simglucose-adolescent2-v0',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={'patient_name': 'adolescent#002',
            'reward_fun': custom_reward}
)

env = gym.make('simglucose-adolescent2-v0')
gamma = 0.99

# Inputs
states = tf.placeholder(tf.float32, shape=(None, 1), name='state')
actions = tf.placeholder(tf.float32, shape=(None,), name='action')
returns = tf.placeholder(tf.int32, shape=(None,), name='return')

# Policy network
pi = tf.keras.layers.Dense(states, [32, 32, env.action_space], name='pi_network')
sampled_actions = tf.squeeze(tf.multinomial(pi, 1))  # For sampling actions according to probabilities.

with tf.variable_scope('pi_optimize'):
    loss_pi = tf.reduce_mean(
        returns * tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=pi, labels=actions), name='loss_pi')
    optim_pi = tf.train.AdamOptimizer(0.001).minimize(loss_pi, name='adam_optim_pi')

# env = gym.make('simglucose-adolescent2-v0')
# gamma = 0.99
# sess = tf.Session(...)

def act(ob):
    return sess.run(sampled_actions, {states: [ob]})

for _ in range(n_episodes):
    ob = env.reset()
    done = False

    obs = []
    actions = []
    rewards = []
    returns = []

    while not done:
        a = act(ob)
        new_ob, r, done, info = env.step(a)

        obs.append(ob)
        actions.append(a)
        rewards.append(r)
        ob = new_ob

    # Estimate returns backwards.
    return_so_far = 0.0
    for r in rewards[::-1]:
        return_so_far = gamma * return_so_far + r
        returns.append(return_so_far)

    returns = returns[::-1]

    # Update the policy network with the data from one episode.
    sess.run([optim_pi], feed_dict={
        states: np.array(obs),
        actions: np.array(actions),
        returns: np.array(returns),
    })



# env = gym.make('simglucose-adolescent2-v0')

# reward = 1
# done = False

# observation = env.reset()
# for t in range(200):
#     env.render(mode='human')
#     # Action in the gym environment is a scalar
#     # representing the basal insulin, which differs from
#     # the regular controller action outside the gym
#     # environment (a tuple (basal, bolus)).
#     # In the perfect situation, the agent should be able
#     # to control the glucose only through basal instead
#     # of asking patient to take bolus
#     action = 0
#     observation, reward, done, info = env.step(action)
#     print(observation)
#     print("Reward = {}".format(reward))
#     if done:
#         print("Episode finished after {} timesteps".format(t + 1))
#         break

# print(env.observation_space.shape[0])