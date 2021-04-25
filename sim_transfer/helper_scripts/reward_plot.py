import matplotlib.pyplot as plt
import numpy as np

def reward(b):
    c0 = 3.35506
    c1 = 0.8353
    c2 = 3.7932
    risk = 10 * (c0 * (np.log(b)**c1 - c2))**2
    return 30-risk  

def emil_reward(b):
    if 20 <= b < 65:
        return 30-(80-65)*3-(65-b)*10
    if 65 <= b < 80:
        return 30-(80-b)*3
    elif 80 <= b < 100:
        return 30
    elif 100 <= b < 140:
        return 30-(b-100)*0.2
    elif 140 <= b:
        return 30-(140-100)*0.2-(b-140)*0.5
    else:
        return 0


f = np.arange(20,400,0.01)
values = []
for item in f:
    values.append(emil_reward(item))

plt.rcParams["figure.figsize"] = (10,10)
plt.plot(f, values, label = "reward")
plt.hlines(0, 0, 400, colors='k', linestyles='dashed', label='', data=None)
plt.ylabel('Reward')
plt.xlabel('Blood Glucose Level [mg/DL]')
plt.legend()
plt.savefig("reward_newest.png", dpi = 100)
