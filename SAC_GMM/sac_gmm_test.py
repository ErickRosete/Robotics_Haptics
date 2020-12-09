import sys
import hydra
from pathlib import Path
from sac_gmm_agent import SAC_GMM_Agent
sys.path.insert(0, str(Path(__file__).parents[1]))
from peg.panda_peg_env import pandaPegV2
from GMM.gmm import GMM
from utils.utils import get_cwd

@hydra.main(config_path="../config", config_name="sac_gmm_config")
def main(cfg):
    env = pandaPegV2(**cfg.train.env)
    model_name =  str((get_cwd() / cfg.gmm_name).resolve())
    gmm_model = GMM(model_name)
    agent = SAC_GMM_Agent(env, gmm_model, **cfg.sac_agent)
    agent.train(**cfg.train.run)

    agent.env = pandaPegV2(**cfg.test.env)
    agent.evaluate(**cfg.test.run)


def evaluate():
    model_name = "models/GMM/gmm_peg_v2_force_3.mat"
    gmm_model = GMM(model_name)
    cfg_env = {"show_gui": False, "with_force": True, "with_joint": False, "relative": True, 
    "with_noise": False ,"sparse": False, "dt": 0.05}
    env = pandaPegV2(**cfg_env)
    agent = SAC_GMM_Agent(env, gmm_model)
    agent.load("peg_290.pth")
    cfg_eval ={"max_steps": 500, "num_episodes": 100, "render": False}   
    accuracy, mean_return, mean_length = agent.evaluate(**cfg_eval)
    print("Accuracy:", accuracy, "Mean return:", mean_return, "Mean length:", mean_length)

if __name__ == "__main__":
    #main()
    evaluate()
   