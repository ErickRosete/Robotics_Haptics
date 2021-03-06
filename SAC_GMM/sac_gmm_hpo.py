import os
import sys
import json 
import hydra
import pickle
import logging
import numpy as np
from pathlib import Path
import ConfigSpace as CS
from sac_gmm_agent import SAC_GMM_Agent
import hpbandster.core.result as hpres
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import ConfigSpace.hyperparameters as CSH
from hpbandster.optimizers import BOHB as BOHB
sys.path.insert(0, str(Path(__file__).parents[1]))
from peg.panda_peg_env import panda_peg_v2
from GMM.gmm import GMM
from utils.utils import get_cwd


class SAC_Worker(Worker):
    def __init__(self, cfg, run_id, nameserver):
        super(SAC_Worker,self).__init__(run_id, nameserver = nameserver)
        self.cfg = cfg
        self.iteration = 0
        self.logger = logging.getLogger(__name__)
    
    def compute(self, config, budget, working_directory, *args, **kwargs):
        env = panda_peg_v2(**self.cfg.env)
        model_name =  str((get_cwd() / self.cfg.gmm_name).resolve())
        gmm_model = GMM(model_name)
        agent = SAC_GMM_Agent(env,gmm_model, **config)

        self.logger.info("Starting agent with budget %d" % budget)
        self.logger.info("Configuration: %s" % json.dumps(config))
        save_dir = "models/iteration_%d" % self.iteration
        self.logger.info("Save directory: %s" % save_dir)
        agent.train(**self.cfg.train, num_episodes = int(budget), save_dir=save_dir)
        accuracy, mean_return, mean_length = agent.evaluate(**self.cfg.validation)
        self.logger.info("Final return reported to the optimizer: %2f, accuracy: %.2f" % (mean_return, accuracy))

        self.iteration += 1
        return ({'loss': - mean_return, # remember: HpBandSter always minimizes!
                 'info': { 'mean_episode_length': mean_length ,
                            'accuracy': accuracy}})
    
    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace()
        actor_lr = CSH.UniformFloatHyperparameter('actor_lr', lower=1e-6, upper=1e-2, log=True)
        critic_lr = CSH.UniformFloatHyperparameter('critic_lr', lower=1e-6, upper=1e-2, log=True)
        alpha_lr = CSH.UniformFloatHyperparameter('alpha_lr', lower=1e-6, upper=1e-2, log=True)
        tau = CSH.UniformFloatHyperparameter('tau', lower=0.001, upper=0.02)
        batch_size = CSH.UniformIntegerHyperparameter('batch_size', lower=128, upper=256)
        hidden_dim = CSH.UniformIntegerHyperparameter('hidden_dim', lower=256, upper=512)
        cs.add_hyperparameters([actor_lr, critic_lr, alpha_lr, tau, batch_size, hidden_dim])
        return cs

def optimize(cfg):
    logger = logging.getLogger(__name__)
    
    NS = hpns.NameServer(run_id=cfg.bohb.run_id, host=cfg.bohb.nameserver, port=None)
    NS.start()

    w = SAC_Worker(cfg.worker, nameserver=cfg.bohb.nameserver, run_id=cfg.bohb.run_id)
    w.run(background=True)


    bohb = BOHB(  configspace = w.get_configspace(),
                run_id = cfg.bohb.run_id, nameserver=cfg.bohb.nameserver,
                min_budget=cfg.bohb.min_budget, max_budget=cfg.bohb.max_budget )
            
    res = bohb.run(n_iterations=cfg.bohb.n_iterations)


    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()

    # Store optimization results
    if not os.path.exists("optimization_results/"): 
        os.makedirs("optimization_results/")
    with open(os.path.join("optimization_results/", "%s.pkl" % cfg.bohb.run_id), 'wb') as fh:
        pickle.dump(res, fh)

    logger.info('Best found configuration:', id2config[incumbent]['config'])
    logger.info('A total of %i unique configurations where sampled.' % len(id2config.keys()))
    logger.info('A total of %i runs where executed.' % len(res.get_all_runs()))
    logger.info('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in res.get_all_runs()])/cfg.bohb.max_budget))
    

@hydra.main(config_path="../config", config_name="sac_gmm_hpo_config")
def main(cfg):
    optimize(cfg)

if __name__ == "__main__":
    main()