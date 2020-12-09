import sys
import hydra
import gym
from pathlib import Path
from sac_agent import SAC_Agent
sys.path.insert(0, str(Path(__file__).parents[1]))
from peg.panda_peg_env import pandaPegV2
import pybullet as p

@hydra.main(config_path="../config", config_name="sac_config")
def main(cfg):
    #env = pandaPegV2(**cfg.train.env)
    env = gym.make("Pendulum-v0")
    agent = SAC_Agent(env, **cfg.agent)
    agent.train(**cfg.train.run)

    # agent.env = pandaPegV2(**cfg.test.env)
    # agent.evaluate(**cfg.test.run)

def evaluate():
    eval_cfg = { "max_steps": 600, "num_episodes": 5, "render": False}
    env_cfg = {"show_gui": True, "with_force": True, 
        "with_joint":False, "relative": True, "with_noise":True, "sparse":False,
        "dt":0.005}
    agent_cfg = {  "batch_size": 256, "gamma": 0.99, "tau": 0.005, "actor_lr": 3e-4,
        "critic_lr": 3e-4, "alpha_lr": 3e-4, "hidden_dim": 256}
    model_name = "outputs/2020-11-24/19-45-10/models/SAC_models/peg_210.pth"
    model_name = "outputs/2020-11-30/00-22-16/models/SAC_models/peg_1060.pth"
    model_name = "outputs/2020-11-30/20-08-23/models/SAC_models/peg_290.pth"

    env = pandaPegV2(**env_cfg)
    agent = SAC_Agent(env, **agent_cfg)
    agent.load(model_name)
    stats = agent.evaluate(**eval_cfg)
    print(stats)

if __name__ == "__main__":

    main()
    #evaluate()