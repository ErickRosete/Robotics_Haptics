from peg.panda_peg_env import  PandaPegEnv
from utils.force_plot import ForcePlot
import numpy as np

# plot = ForcePlot()

env = PandaPegEnv()
error = 0.001
dt = 1./240. # the default timestep in pybullet is 240 Hz  
target = []
target_position = [0.7, 0, 0.1]
demo = 1

for i_episode in range(100):
    observation = env.reset()
    exp, state = [], 0 
    k_p, k_d = 10, 0.1
    for t in range(2000):
        if state == 0:
            target = [target_position[0] - 0.125, target_position[1], target_position[2] + 0.07]
        elif state == 1:
            k_d = 0.05
            target = [target_position[0], target_position[1], target_position[2] + 0.063]
        elif state == 2:
            target = [target_position[0], target_position[1], target_position[2] + 0.0175]

        dx = target[0] - observation[0]
        dy = target[1] - observation[1]
        dz = target[2] - observation[2]
        pd_x = k_p*dx + k_d*dx/dt
        pd_y = k_p*dy + k_d*dy/dt
        pd_z = k_p*dz + k_d*dz/dt
        action = np.array([pd_x, pd_y, pd_z, -0.4])

        #Save data
        data = observation.tolist()
        data = data[:3] + data[-2:]
        data.insert(0, (t * dt))
        exp.append(data)

        observation, reward, done, info = env.step(action)
        target_position = info['target_position']

         #Force measurements
        # if t % 20 == 0:
        #     plot.update(observation[5:])
        if abs(dx) < error and abs(dy) < error and abs(dz) < error:
            state += 1

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            # End of opening drawer
            name = "demonstrations/peg_v2/peg_v2_%d.txt" % (demo)
            exp = np.stack(exp, axis =0)
            np.savetxt(name, exp)
            demo += 1

            break
env.close()