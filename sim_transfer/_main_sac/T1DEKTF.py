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

#Patient Parameters for 10 children, adolescents and adults
PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')


##---------------------######################
# Type 1 Diabetes Environment 

class T1DEKTF(py_environment.PyEnvironment):
    """
        Custom Environment that follows TF-agents interface
    """ 
    
    SENSOR_HARDWARE = 'Dexcom'
    INSULIN_PUMP_HARDWARE = 'Insulet'
    

    def __init__(self, patient_name=None, reward_function=None, seed=None):
        
        state_length = 120 #in minutes, passed to rnn
        
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(state_length,), dtype=np.float32, minimum=0, maximum=np.inf, name='simulator')
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.float32, minimum=0, maximum=0.09, name='insulin')
        
        self._state =  [0] * state_length
        self._state = np.asarray(self._state).reshape(self._observation_spec.shape)
        self._episode_ended = False

        if patient_name is None:
            patient_name = 'adolescent#001' #Default patient if none selected 
        self.patient_name = patient_name
        self.reward_function = reward_function
        self.np_random, _ = seeding.np_random(seed=seed)
        self.simulator, _, _, _ = self.create_simulator_from_random_state()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _step(self, action):
        """ 
            New action affects the simulator, first check if simultor is still running
        """
        
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
        
        #Insert new observation into state
        self._state = np.delete(self._state, 0, axis=0)
        self._state = np.insert(self._state, self._observation_spec.shape[0]-1, obs, axis=0)
        self._state = np.reshape(self._state, self._observation_spec.shape)

        self._episode_ended = done 

        if done:
            return ts.termination(np.array(self._state, dtype=np.float32), reward)
        else:
            return ts.transition(np.array(self._state, dtype=np.float32), reward=reward, discount=1.0)


    def seed(self, seed=None):
        """
            Seed function for simulator
        """
        
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3, seed4 = self.create_simulator_from_random_state()
        return [seed1, seed2, seed3, seed4]

    def create_simulator_from_random_state(self):
        """
            Create new simulator instance 
        """ 
        
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

        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        simulator = _T1DSimEnv(patient, sensor, pump, scenario)
        return simulator, seed2, seed3, seed4

    def _reset(self):
        """
            Reset simulator to defaults
        """
        self.simulator, _, _, _ = self.create_simulator_from_random_state()
        obs, _, _, _ = self.simulator.reset()
        obs = obs[0]
        self._episode_ended = False
        
        self._state =  [0] * state_length
        self._state = np.delete(self._state, 0, axis=0)
        self._state = np.insert(self._state, self._observation_spec.shape[0]-1, obs, axis=0)
        self._state = np.reshape(self._state, self._observation_spec.shape)
        return ts.restart(np.array(self._state, dtype=np.float32))

    def render(self, filename="test.png"):
        """
            Render the episode to png
        """
        normal_bg, low_bg, high_bg = self.simulator.render(close=False)
        plt = self.simulator.viewer.fig
        plt.savefig(filename)
        self.simulator.viewer.close()
        return normal_bg, low_bg, high_bg
