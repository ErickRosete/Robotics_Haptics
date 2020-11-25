from peg.panda_peg_env import  PandaPegEnv
from utils.force_plot import ForcePlot
import numpy as np

plot = ForcePlot()

env = PandaPegEnv()
error = 0.001
k_p = 10
k_d = 0.1
dt = 1./240. # the default timestep in pybullet is 240 Hz  
target = []
target_position = [0.7, 0, 0.1]

for i_episode in range(20):
    observation = env.reset()
    state = 0
    for t in range(2000):
        if state == 0:
            target = [target_position[0]+0.05, target_position[1], target_position[2] + 0.05, 0.024]
        elif state == 1:
            target = [target_position[0]+0.05, target_position[1], target_position[2] + 0.03,  0.024]
        elif state == 2:
            target = [target_position[0], target_position[1], target_position[2] + 0.03, 0.04]

        dx = target[0] - observation[0]
        dy = target[1] - observation[1]
        dz = target[2] - observation[2]
        df = target[3] - (observation[3] + observation[4])/2
        pd_x = k_p*dx + k_d*dx/dt
        pd_y = k_p*dy + k_d*dy/dt
        pd_z = k_p*dz + k_d*dz/dt
        pd_f = k_p*df + k_d*df/dt
        action = np.array([pd_x,pd_y,pd_z,pd_f])
        observation, reward, done, info = env.step(action)
        target_position = info['target_position']
        
        #Force measurements
        if t % 20 == 0:
            plot.update(observation[5:])
        if abs(dx) < error and abs(dy) < error and abs(dz) < error and abs(df) < 0.003:
            state += 1

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()