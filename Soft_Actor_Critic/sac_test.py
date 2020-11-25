import sys
import hydra
from pathlib import Path
from sac_agent import SAC_Agent
sys.path.insert(0, str(Path(__file__).parents[1]))
from peg.panda_peg_env import pandaPegV2
import pybullet as p

@hydra.main(config_path="../config", config_name="sac_config")
def main(cfg):
    env = pandaPegV2(**cfg.train.env)
    agent = SAC_Agent(env, **cfg.agent)
    agent.train(**cfg.train.run)

    agent.env = pandaPegV2(**cfg.test.env)
    agent.evaluate(**cfg.test.run)

def evaluate():
    model = 3

    eval_cfg = { "max_steps": 2000, "num_episodes": 10, "render": False}
    if model == 1:
        env_cfg = {"show_gui": True, "withForce": True, 
        "withJoint":False, "relative": False, "noise":False, "sparse":False,
        "dt":0.005}
        agent_cfg = {  "batch_size": 256, "gamma": 0.99, "tau": 0.005, "actor_lr": 3e-4,
        "critic_lr": 3e-4, "alpha_lr": 3e-4, "hidden_dim": 256}

        env = pandaPegV2(**env_cfg)
        agent = SAC_Agent(env, **agent_cfg)
        agent.load("outputs/2020-11-24/19-45-10/models/SAC_models/peg_210.pth")
    else:
        env_cfg = {"show_gui": True, "withForce": True, 
        "withJoint":False, "relative": True, "noise":False, "sparse":False,
        "dt":0.005}
        agent_cfg = {  "batch_size": 256, "gamma": 0.99, "tau": 0.005, "actor_lr": 3e-4,
        "critic_lr": 3e-4, "alpha_lr": 3e-4, "hidden_dim": 400}
        env = pandaPegV2(**env_cfg)
        agent = SAC_Agent(env, **agent_cfg)
        if model == 2:
            agent.load("outputs/2020-11-25/07-46-41/models/SAC_models/peg_430.pth")
        else:
            agent.load("outputs/2020-11-25/13-12-53/models/SAC_models/peg_200.pth")
    val_return = agent.evaluate(**eval_cfg)
    print(val_return)

if __name__ == "__main__":
    main()
    #evaluate()