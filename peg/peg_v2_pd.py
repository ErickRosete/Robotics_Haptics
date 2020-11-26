import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from peg.panda_peg_env import  PandaPegEnv
from utils.force_plot import ForcePlot
import numpy as np

show_force = False
show_gui = True

if show_force:
    plot = ForcePlot()

env = PandaPegEnv(show_gui)
error = 0.0015
dt = 1./240. # the default timestep in pybullet is 240 Hz  
target = []
target_position = [0.7, 0, 0.1]

for i_episode in range(20):
    observation = env.reset()
    state = 0
    k_p, k_d = 5, 0.05
    for t in range(2000):
        if state == 0:
            target = [target_position[0] - 0.125, target_position[1], target_position[2] + 0.07]
        elif state == 1:
            target = [target_position[0], target_position[1], target_position[2] + 0.063]
        elif state == 2:
            target = [target_position[0], target_position[1], target_position[2] + 0.0175]

        dx = target[0] - observation[0]
        dy = target[1] - observation[1]
        dz = target[2] - observation[2]
        pd_x = k_p*dx + k_d*dx/dt
        pd_y = k_p*dy + k_d*dy/dt
        pd_z = k_p*dz + k_d*dz/dt
        action = np.array([pd_x,pd_y,pd_z, -0.4])
        observation, reward, done, info = env.step(action)
        target_position = info['target_position']
        
        #Force measurements
        if show_force and t % 20 == 0:
            plot.update(observation[5:])

        if abs(dx) < error and abs(dy) < error and abs(dz) < error:
            state += 1
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()