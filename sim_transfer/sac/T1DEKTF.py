import tensorflow as tf
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts


from simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.scenario import CustomScenario
from simglucose.controller.base import Action
import numpy as np
import pkg_resources
import gym
from gym import spaces
from gym.utils import seeding
from datetime import datetime
import matplotlib.pyplot as plt

PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')

#Formula: BW*u2ss/6000=BR

#Adolescent1: BW 68.706, u2ss 1.2169            -> BR = 0.014 ->max = 7 * BR = 0.098 #0.09
#Adult7: BW 91.229, u2ss 1.503                  -> BR = 0.02285 -> max = 7 *BR = 0.1599 #0.1?
#Adolescent9: BW 43.885, u2ss 1.38186522046     -> BR = 0.0101 -> max = 0.0708 
#Child 4: BW 35.5165043, u2ss 1.38610897835     -> BR = 8.204*10e-3 -> 0.0574
#Child7 = 0.0574

class T1DEKTF(py_environment.PyEnvironment):
    """Custom Environment that follows TF interface""" 
    SENSOR_HARDWARE = 'Dexcom'
    INSULIN_PUMP_HARDWARE = 'Insulet'

    def __init__(self, patient_name=None, reward_function=None, seed=None):
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(120,), dtype=np.float32, minimum=0, maximum=np.inf, name='simulator')
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.float32, minimum=0, maximum=0.09, name='insulin')
        self._state =  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        #self._state =  [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
        self._state = np.asarray(self._state).reshape(self._observation_spec.shape)
        self._episode_ended = False

        if patient_name is None:
            patient_name = 'adolescent#001' #Change patient here 
        self.patient_name = patient_name
        self.reward_function = reward_function
        self.np_random, _ = seeding.np_random(seed=seed)
        self.simulator, _, _, _ = self.create_simulator_from_random_state()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _step(self, action):
        if self._episode_ended:
            return self.reset()
        else:
            pass

        act = Action(basal=action, bolus=0)
        if self.reward_function is None:
            obs, reward, done, _ = self.simulator.step(act)
        else:
            obs, reward, done, _ = self.simulator.step(act, reward_fun=self.reward_function)
        obs = obs[0]
        # self._state = np.delete(self._state, 0)
        # self._state = np.insert(self._state, self._observation_spec.shape[0]-1, obs, axis=0)
        self._state = np.delete(self._state, 0, axis=0)
        self._state = np.insert(self._state, self._observation_spec.shape[0]-1, obs, axis=0)
        self._state = np.reshape(self._state, self._observation_spec.shape)

        self._episode_ended = done 

        if done:
            return ts.termination(np.array(self._state, dtype=np.float32), reward)
        else:
            return ts.transition(np.array(self._state, dtype=np.float32), reward=reward, discount=1.0)


    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3, seed4 = self.create_simulator_from_random_state()
        return [seed1, seed2, seed3, seed4]

    def create_simulator_from_random_state(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(self.np_random.randint(0, 1000)) % 2**31
        seed3 = seeding.hash_seed(seed2 + 1) % 2**31
        seed4 = seeding.hash_seed(seed3 + 1) % 2**31

        hour = self.np_random.randint(low=0.0, high=24.0)
        ## For starting at the same hour
        #hour = 0

        start_time = datetime(2020, 1, 1, hour, 0, 0)
        patient = T1DPatient.withName(self.patient_name, random_init_bg=True, seed=seed4)
        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)

        scenario = RandomScenario(start_time=start_time, seed=seed3)
        ## This does not work yet, only with the function RandomScenario sadly
        #scen = [(7, 45), (12, 70), (16, 15), (18, 80)]
        #scenario = CustomScenario(start_time=start_time, scenario=scen)

        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        simulator = _T1DSimEnv(patient, sensor, pump, scenario)
        return simulator, seed2, seed3, seed4

    def _reset(self):
        self.simulator, _, _, _ = self.create_simulator_from_random_state()
        obs, _, _, _ = self.simulator.reset()
        obs = obs[0]
        self._episode_ended = False
        self._state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, obs]
        #self._state = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[obs,0]]
        self._state = np.asarray(self._state).reshape(self._observation_spec.shape)
        return ts.restart(np.array(self._state, dtype=np.float32))

    def render(self, filename="test.png"):
        normal_bg, low_bg, high_bg = self.simulator.render(close=False)
        plt = self.simulator.viewer.fig
        plt.savefig(filename)
        self.simulator.viewer.close()
        return normal_bg, low_bg, high_bg
