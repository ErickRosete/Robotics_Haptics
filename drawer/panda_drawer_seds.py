from panda_drawer_env import  PandaDrawerEnv
import matlab.engine
import numpy as np

#Start matlab
eng = matlab.engine.start_matlab()

# Generate environment
env = PandaDrawerEnv()

for i_episode in range(20):
    object_position, target = [0.45, -0.023, 0.15], 0
    model = 0
    observation = env.reset()
    for t in range(2000):
        if model == 0:
            target = [object_position[0], object_position[1], object_position[2], 0.01, 0.01] 
        elif model == 1:
            target = [0.23, object_position[1], object_position[2], 0.001, 0.01]

        #Call seds model to predict next action
        x0 = matlab.double(observation[:5].tolist()) #x,y,z,f1,f2
        xT = matlab.double(target)
        vel = np.asarray(eng.predict_vel(x0, xT, model)).squeeze()
        action = np.array( [vel[0], vel[1], vel[2], (vel[3] + vel[4])/2] )
        observation, reward, done, info = env.step(action, dt=0.05)
        object_position = info['object_position']

        # Converges to target
        if(np.linalg.norm(vel) < 0.008):
            model = 1

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
eng.quit()
