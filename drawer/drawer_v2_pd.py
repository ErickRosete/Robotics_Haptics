from drawer.panda_drawer_env import  PandaDrawerEnv
from utils.force_plot import ForcePlot
import numpy as np


# Only move xyz to guide to the appropiate position.

front, open = True, True
env = PandaDrawerEnv(front, open)
error = 0.01
dt = 1./240. # the default timestep in pybullet is 240 Hz  
target = []
object_position = [0.7, 0, 0.1]
# plot = ForcePlot()

for i_episode in range(20):
    observation = env.reset()
    state = 0
    k_p, k_d = 10, 0.1
    for t in range(2000):
        if state == 0:
            if front:
                target = [object_position[0] - 0.01, object_position[1], object_position[2] + 0.20]
            else:
                target = [object_position[0], object_position[1], object_position[2] + 0.05]
        elif state == 1:
            k_d = 0.05
            target = [object_position[0], object_position[1], object_position[2]]
        elif state == 2:
            k_d = 0.02
            target = [0.25, object_position[1], object_position[2]]

        dx = target[0] - observation[0]
        dy = target[1] - observation[1]
        dz = target[2] - observation[2]
        pd_x = k_p*dx + k_d*dx/dt
        pd_y = k_p*dy + k_d*dy/dt
        pd_z = k_p*dz + k_d*dz/dt
        action = np.array([pd_x,pd_y,pd_z,0])

        observation, reward, done, info = env.step(action)
        object_position = info['object_position']
        
        #Force measurements
        # if t % 30 == 0:
        #     plot.update(observation[5:])

        if abs(dx) < error and abs(dy) < error and abs(dz) < error:
            state += 1
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()