from peg.panda_peg_env import  PandaPegEnv
import matlab.engine
import numpy as np

#Start matlab
eng = matlab.engine.start_matlab()

# Generate environment
env = PandaPegEnv()

for i_episode in range(20):
    target_position, target = [0.45, -0.023, 0.15], 0
    observation = env.reset()
    for t in range(1000):
        target = [target_position[0], target_position[1], target_position[2] + 0.03, 0.04, 0.04]

        #Call seds model to predict next action
        x0 = matlab.double(observation[:5].tolist()) #x,y,z,f1,f2
        xT = matlab.double(target)
        vel = np.asarray(eng.peg_predict_vel(x0, xT)).squeeze()
        action = np.array( [vel[0], vel[1], vel[2], (vel[3] + vel[4])/2] )
        observation, reward, done, info = env.step(action, dt=0.05)
        target_position = info['target_position']

        # Converges to target
        # if(model == 0 and np.linalg.norm(vel) < 0.008):
        #     model = 1
            
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
eng.quit()
