import sys
import hydra
import gym
import os
from pathlib import Path
from sac_agent import SAC_Agent
sys.path.insert(0, str(Path(__file__).parents[1]))
from peg.panda_peg_env import panda_peg_v2
import gym

def get_save_filename(cfg, it=0):
    noise = 'noise_' if cfg.train.env.with_noise else ''
    force = "force" if cfg.train.env.with_force else "pose"
    rs = "_rs_" + str(it) if hasattr(cfg.train, 'num_random_seeds') and cfg.train.num_random_seeds > 1 else ''
    save_filename = "sac_peg_v2_" + noise + force  + rs
    return save_filename


@hydra.main(config_path="../config", config_name="sac_config")
def main(cfg):
    for i in range(cfg.train.num_random_seeds):
        # Training 
        #env = panda_peg_v2(**cfg.train.env)
        env = gym.make('Pendulum-v0')
        env.max_episode_steps = cfg.train.env.max_episode_steps
        agent = SAC_Agent(env, **cfg.agent)
        save_filename = get_save_filename(cfg, i)
        agent.train(**cfg.train.run, save_filename=save_filename)
        agent.env.close()

        # Testing
        # agent.env = panda_peg_v2(**cfg.test.env)
        # agent.evaluate(**cfg.test.run)
        # agent.env.close()

def evaluate():
    eval_cfg = {"num_episodes": 20, "render": False}
    env_cfg = {"show_gui": True, "max_episode_steps": 500, "with_force": False, 
        "with_joint":False, "relative": True, "with_noise":False, "reward_type":"shaped_2",
        "dt":0.005}
    agent_cfg = {  "batch_size": 256, "gamma": 0.99, "tau": 0.005, "actor_lr": 3e-4,
        "critic_lr": 3e-4, "alpha_lr": 3e-4, "hidden_dim": 256}
    model_name = "sac_peg_v2_pose_2200.pth"

    env = panda_peg_v2(**env_cfg)
    agent = SAC_Agent(env, **agent_cfg)
    agent.load(model_name)
    stats = agent.evaluate(**eval_cfg)
    print(stats)

if __name__ == "__main__":
    #main()
    evaluate()