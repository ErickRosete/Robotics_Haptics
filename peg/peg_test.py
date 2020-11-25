from utils.force_plot import ForcePlot
from peg.panda_peg_env import PandaPegEnv

if __name__ == "__main__":
    env = PandaPegEnv()
    # plot = ForcePlot()

    for episode in range(100):
        print(episode)
        s = env.reset()
        episode_length, episode_reward = 0,0
        for step in range(500):
            a = env.action_space.sample()
            a[3] = -0.05
            s, r, done, _ = env.step(a)
            # img = env.render()
            # print(img.shape)

            #Update Plot
            # plot.update(s[5:])

            if(done):
                break
