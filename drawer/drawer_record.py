from drawer.panda_drawer_env import  PandaDrawerEnv
from utils.force_plot import ForcePlot
import numpy as np

# plot = ForcePlot()

front = False
env = PandaDrawerEnv(front)
error = 0.01
k_p = 10
k_d = 0.1
dt = 1./240. # the default timestep in pybullet is 240 Hz  
target = []
object_position = [0.7, 0, 0.1]
exp = []

for i_episode in range(15):
    observation = env.reset()
    state = 0
    for t in range(3000):
        if state == 0:
            if front:
                target = [object_position[0] - 0.01, object_position[1], object_position[2] + 0.20, 0.04]
            else:
                target = [object_position[0], object_position[1], object_position[2] + 0.05, 0.04]
        elif state == 1:
            target = [object_position[0], object_position[1], object_position[2],  0.04]
        elif state == 2:
            target = [object_position[0], object_position[1], object_position[2], 0.01]
        elif state == 3:
            k_d = 0.01
            target = [0.20, object_position[1], object_position[2], 0.01]

        dx = target[0] - observation[0]
        dy = target[1] - observation[1]
        dz = target[2] - observation[2]
        df = target[3] - (observation[3] + observation[4])/2
        pd_x = k_p*dx + k_d*dx/dt
        pd_y = k_p*dy + k_d*dy/dt
        pd_z = k_p*dz + k_d*dz/dt
        pd_f = k_p*df + k_d*df/dt
        action = np.array([pd_x,pd_y,pd_z,pd_f])
        
        #Save data
        data = observation.tolist()
        data.insert(0, (t * dt))
        exp.append(data)

        observation, reward, done, info = env.step(action)
        object_position = info['object_position']

        #Force measurements
        # if t % 20 == 0:
        #     plot.update(observation[5:])

        if abs(dx) < error and abs(dy) < error and abs(dz) < error and abs(df) < error:
            if(state == 2): 
                force = (observation[5] + observation[6])/2
                if force > 0.25: #Also check force
                    if not front:
                        exp = np.stack(exp, axis =0)
                        np.savetxt(("demonstrations/grabbing_%d.txt" % (i_episode + 1)), exp)
                        exp = []
                    state = 3
            else:
                state += 1

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            # End of opening drawer
            name = "demonstrations/opening_%d.txt" % (i_episode + 1)
            if front:
                name = "demonstrations/drawer_%d.txt" % (i_episode + 1)
            exp = np.stack(exp, axis =0)
            np.savetxt(name, exp)
            exp = []

            break
env.close()