# from panda_peg_env import  PandaPegEnv
# import matlab.engine
# import numpy as np
# from utils.force_plot import ForcePlot


# def evaluate(num_episodes=10, mode="pose", model_name="gmm_peg_v2_3.mat", show_force=True, noise=False, vel_factor=10):
#     succesful_episodes = 0

#     if show_force:
#         plot = ForcePlot()

#     #Start matlab
#     eng = matlab.engine.start_matlab()

#     # Generate environment
#     env = PandaPegEnv(show_gui=True)
#     for i_episode in range(num_episodes):
#         target_position, target = [0.45, -0.023, 0.15], 0
#         observation = env.reset()
#         for t in range(2000):
#             if t <= 1:
#                 target = [target_position[0], target_position[1], target_position[2] + 0.0175]
#                 if noise:
#                     target = (np.array(target) + np.random.normal(0, 0.01)).tolist()
#                 if mode == "force":
#                     target += [0.6, 0.6]
#             #Call GMM+GMR model to predict next action
#             data = observation.tolist()
#             if mode == "force":
#                 data = data[:3] + data[-2:]
#             else:
#                 data = data[:3]    
#             x0 = matlab.double(data) #x,y,z,f1,f2
#             xT = matlab.double(target)
#             vel = np.asarray(eng.predict_vel(model_name, x0, xT)).squeeze()
#             action = np.array( [vel[0], vel[1], vel[2], -0.4/vel_factor] )
#             observation, reward, done, info = env.step(action, dt=0.005*vel_factor)
#             target_position = info['target_position']

#             if show_force and t % 20 == 0:
#                 plot.update(observation[5:])

#             if done:
#                 print("Episode finished after {} timesteps".format(t+1))
#                 succesful_episodes += 1
#                 break
#     env.close()
#     eng.quit()
#     return succesful_episodes

# if __name__ == "__main__":
#     num_episodes = 50
#     mode = "pose"
#     show_force = False
#     noise = False

#     if mode == "pose":
#         model_name = "models/GMM_models/gmm_peg_v2_pose_9.mat"
#         vel_factor = 10
#         succesful_episodes = evaluate(num_episodes, mode, model_name, show_force, noise, vel_factor)  
#         print("%s accuracy: %2f" % (model_name, succesful_episodes/num_episodes))
#         #Best so far 9 = 0.9
#     else:
#         vel_factor = 10
#         # for i in range(13, 16):
#         i=15
#         model_name = 'gmm_peg_v2_%d.mat' % i
#         #model_name = "GMM_models/gmm_peg_v2_11.mat"
#         succesful_episodes = evaluate(num_episodes, mode, model_name, show_force, noise, vel_factor)  
#         print("========================================")
#         print("%s accuracy: %2f" % (model_name, succesful_episodes/num_episodes))
#         #Best so far 15 = 0.6 
