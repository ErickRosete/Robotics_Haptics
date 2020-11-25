from panda_drawer_env import  PandaDrawerEnv

env = PandaDrawerEnv()
for episode in range(100):
    print(episode)
    s = env.reset()
    episode_length, episode_reward = 0,0
    for step in range(500):
        a = env.action_space.sample()
        s, r, done, _ = env.step(a)
        img = env.render()
        if(done):
            break
