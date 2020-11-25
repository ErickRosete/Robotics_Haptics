from drawer.panda_drawer_env import  PandaDrawerEnv
import matlab.engine
import numpy as np
from utils.force_plot import ForcePlot

def evaluate(num_episodes=10, mode="pose", model_name="gmm_drawer_v2_3.mat", show_force=True):
    succesful_episodes = 0

    if show_force:
        plot = ForcePlot()

    #Start matlab
    eng = matlab.engine.start_matlab()

    # Generate environment
    front, open = True, True
    env = PandaDrawerEnv(front, open)

    for i_episode in range(num_episodes):
        object_position, target = [0, 0, 0], 0
        observation = env.reset()
        for t in range(2000):
            if t <= 1:
                target = [0.25, object_position[1], object_position[2]] 
                if mode == "force":
                    target += [0.02]
            #Call GMM+GMR model to predict next action
            data = observation.tolist()
            if mode == "force":
                data = data[:3] + data[-1:]
            else:
                data = data[:3]          
            x0 = matlab.double(data) #x,y,z,f1,f2
            xT = matlab.double(target)
            vel = np.asarray(eng.predict_vel(model_name, x0, xT)).squeeze()
            action = np.array( [vel[0], vel[1], vel[2], 0] )
            observation, reward, done, info = env.step(action, dt=0.05)
            object_position = info['object_position']
            
            if show_force and t % 20 == 0:
                plot.update(observation[5:])
                
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                succesful_episodes += 1
                break
    env.close()
    eng.quit()
    return succesful_episodes

if __name__ == "__main__":
    num_episodes = 10
    mode = "pose"
    show_force = True

    if mode == "pose":
        model_name = 'GMM_models/gmm_drawer_v2_pose_9.mat'
    else:
        model_name = 'GMM_models/gmm_drawer_v2_9.mat'

    succesful_episodes = evaluate(num_episodes, mode, model_name, show_force)  
    print("%s accuracy: %2f" % (model_name, succesful_episodes/num_episodes))      