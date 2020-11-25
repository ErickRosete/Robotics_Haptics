from drawer.panda_drawer_env import  PandaDrawerEnv
import matlab.engine
import numpy as np
from utils.force_plot import ForcePlot

def evaluate(num_episodes=10, mode="pose", model_names=["seds_drawer_3.mat"], show_force=True, front=True):
    succesful_episodes = 0

    if show_force:
        plot = ForcePlot()

    #Start matlab
    eng = matlab.engine.start_matlab()

    # Generate environment
    env = PandaDrawerEnv(front)

    for i_episode in range(num_episodes):
        object_position, target = [0.45, -0.023, 0.15], 0
        model_idx = 0

        observation = env.reset()
        for t in range(2000):
            if front and model_idx == 0:
                target = [object_position[0], object_position[1], object_position[2], 0.01, 0.01] 
            else:
                target = [0.26, object_position[1], object_position[2], 0.01, 0.01]


            #Call seds model to predict next action
            x0 = matlab.double(observation[:5].tolist()) #x,y,z,f1,f2
            xT = matlab.double(target)
            vel = np.asarray(eng.drawer_predict_vel(x0, xT, model_names[model_idx])).squeeze()
            action = np.array( [vel[0], vel[1], vel[2], (vel[3] + vel[4])/2] )
            observation, reward, done, info = env.step(action, dt=0.05)
            object_position = info['object_position']

            # Converges to target
            # if(model == 0 and np.linalg.norm(vel) < 0.008):
            #     model = 1
            
            force = (observation[5] + observation[6])/2
            if model_idx == 0 and force > 0.25: #Also check force
                model_idx = 1
            
            if show_force and t % 20 == 0:
                plot.update(observation[5:])

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()
    eng.quit()
    return succesful_episodes

if __name__ == "__main__":
    num_episodes = 10
    mode = "pose"
    show_force = True
    front = True
    if front:
        model_names = ['SEDS_models/seds_drawer_1000_mse_6.mat']
    else:
        model_names = ["SEDS_models/seds_grabbing_1000_mse_5.mat", "SEDS_models/seds_opening_1500_mse_6.mat"]
    succesful_episodes = evaluate(num_episodes, mode, model_names, show_force, front)  
    print("%s accuracy: %2f" % (model_names[0], succesful_episodes/num_episodes))      