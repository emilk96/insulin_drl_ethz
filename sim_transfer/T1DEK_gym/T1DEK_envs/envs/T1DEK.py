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

#Parameters of patients blood glucose dynamics
PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')

class T1DEK(gym.Env):
  """Custom Environment that follows gym interface
  """
  metadata = {'render.modes': ['rgb']}

  SENSOR_HARDWARE = 'Dexcom'
  INSULIN_PUMP_HARDWARE = 'Insulet'


  def __init__(self, patient_name=None, reward_function=None, seed=None):
    super(T1DEK, self).__init__()
    if patient_name is None:
        patient_name = 'adolescent#001'
    self.patient_name = patient_name
    self.reward_function = reward_function
    self.np_random, _ = seeding.np_random(seed=seed)
    self.simulator, _, _, _ = self.create_simulator_from_random_state()

    ub = self.simulator.pump._params['max_basal']
    self.action_space = spaces.Box(low=0, high=0.1, shape=(1,))
    self.observation_space = spaces.Box(low=0, high=np.inf, shape=(1,))
    ##For incorporating meals
    #self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2,))


  def step(self, action):
    act = Action(basal=action, bolus=0)

    if self.reward_function is None:
        return self.simulator.step(act)
    else:
        return self.simulator.step(act, reward_fun=self.reward_function)

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

  def reset(self):
    self.simulator, _, _, _ = self.create_simulator_from_random_state()
    obs, _, _, _ = self.simulator.reset()
    return obs

  def render(self, filename="test.png"):
    self.simulator.render(close=False)
    plt = self.simulator.viewer.fig
    plt.savefig(filename)
    self.simulator.viewer.close()


  # def render_update(self, mode='rgb', close=False):
  #   # Render the environment to the screen
  #   self.simulator.render(close=close)

  # def save_renders(self, filename="test.png"):
  #   plt = self.simulator.viewer.fig
  #   plt.savefig(filename)
  #   self.simulator.viewer.close()
    
