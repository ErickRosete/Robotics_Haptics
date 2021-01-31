import numpy as np
from peg_v2_pd import PegPD
from panda_peg_env import  panda_peg_v2

if __name__ == "__main__":
    n_demo = 1
    dt = 0.005
    env_config = { "show_gui":False, "dt": dt, "reward_type": "shaped_2", 
                "with_force":True, "with_joint":False, "relative":False,
                "with_noise":False}
    env = panda_peg_v2(**env_config)
    pd = PegPD(env)
    num_episodes = 100

    for i_episode in range(num_episodes):
        episode_return = 0
        observation = env.reset()
        pd.reset()
        exp = []
        for t in range(env.max_episode_steps):
            action = pd.get_action()
            observation, reward, done, info = env.step(action)
            episode_return += reward

            #Save data
            data = observation.tolist()
            data.insert(0, (t * dt))
            exp.append(data)

            if done:
                if info['success']:
                    print("Episode finished after {} timesteps".format(t+1))
                    name = "demonstrations/peg_v2/new_peg_v2_%d.txt" % (n_demo)
                    exp = np.stack(exp, axis =0)
                    np.savetxt(name, exp)
                    n_demo += 1
                    break
    env.close()