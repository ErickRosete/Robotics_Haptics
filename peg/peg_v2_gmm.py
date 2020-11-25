import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from panda_peg_env import  pandaPegV2
import matlab.engine
import numpy as np
from utils.force_plot import ForcePlot

def evaluate(env, model_name="gmm_peg_v2_3.mat", num_episodes=10, show_force=True):
    succesful_episodes = 0

    if show_force:
        plot = ForcePlot()

    #Start matlab
    eng = matlab.engine.start_matlab()

    for i_episode in range(num_episodes):
        target_position, target = [0.45, -0.023, 0.15], 0
        observation = env.reset()
        for t in range(2000):
            x = matlab.double( observation.tolist() ) #x,y,z,f1,f2
            vel = np.asarray(eng.predict_rel_vel(model_name, x)).squeeze()
            observation, reward, done, info = env.step(vel[:3])
            if show_force and t % 20 == 0:
                plot.update(observation[5:])
            if done:
                break
        if info['success']:
            print("Succesful episode finished after {} timesteps".format(t+1))
            succesful_episodes += 1
    eng.quit()
    return succesful_episodes

if __name__ == "__main__":

    # Environment hyperparameters
    show_gui = False
    withForce = True
    withJoint = False
    relative = True
    noise = False
    dt = 0.05
    env = pandaPegV2(show_gui, dt, withForce, withJoint, relative, noise)

    # Evaluate parameters
    num_episodes = 10
    show_force = False
    K_range = [8, 18]
    best_model, best_acc = "", 0
    for i in range(K_range[0], K_range[1]):
        model_name = "gmm_peg_v2_%d.mat" % i
        succesful_episodes = evaluate(env, model_name, num_episodes, show_force)  
        accuracy = succesful_episodes/num_episodes
        print("%s accuracy: %2f" % (model_name, accuracy ))
        if accuracy > best_acc:
            best_acc = accuracy
            best_model = model_name
    
    print("The best model is: %s  with accuracy: %2f" % (best_model, best_acc ))
