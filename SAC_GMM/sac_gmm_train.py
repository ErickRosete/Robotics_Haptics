import os
import sys
import hydra
from pathlib import Path
from sac_gmm_agent import SAC_GMM_Agent
sys.path.insert(0, str(Path(__file__).parents[1]))
from peg.panda_peg_env import panda_peg_v2
from GMM.gmm import GMM
from utils.utils import get_cwd


def get_save_filename(gmm_name, cfg, it=0):
    base = os.path.basename(gmm_name)
    name_postfix = os.path.splitext(base)[0][4:]
    noise = 'noise_' if cfg.train.env.with_noise else ''
    force = "force_" if cfg.train.env.with_force else ""
    rs = "_rs_" + str(it) if hasattr(cfg.train, 'num_random_seeds') and cfg.train.num_random_seeds > 1 else ''
    save_filename = "sac_gmm_" + noise + force + name_postfix  + rs
    return save_filename

@hydra.main(config_path="../config", config_name="sac_gmm_config")
def main(cfg):
    for i in range(cfg.train.num_random_seeds):
        # Training 
        env = panda_peg_v2(**cfg.train.env)
        model_name =  str((get_cwd() / cfg.gmm_name).resolve())
        gmm_model = GMM(model_name)
        agent = SAC_GMM_Agent(env, gmm_model, **cfg.sac_agent)
        save_filename = get_save_filename(cfg.gmm_name, cfg, i)
        agent.train(**cfg.train.run, save_filename=save_filename)
        agent.env.close()

        # Testing
        agent.env = panda_peg_v2(**cfg.test.env)
        agent.evaluate(**cfg.test.run)
        agent.env.close()

@hydra.main(config_path="../config", config_name="mult_sac_gmm_config")
def mult_sac_gmm(cfg):
    for gmm_name in cfg.gmm_names:
        env = panda_peg_v2(**cfg.train.env)

        model_name =  str((get_cwd() / gmm_name).resolve())
        gmm_model = GMM(model_name)
        agent = SAC_GMM_Agent(env, gmm_model, **cfg.sac_agent)
        save_filename = get_save_filename(gmm_name, cfg)
        
        # Training
        agent.train(**cfg.train.run, save_filename=save_filename)
        agent.env.close()

        # Testing
        agent.env = panda_peg_v2(**cfg.test.env)
        agent.evaluate(**cfg.test.run)
        agent.env.close()

if __name__ == "__main__":
    main()
    #mult_sac_gmm()