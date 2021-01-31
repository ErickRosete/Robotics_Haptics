import sys
from pathlib import Path
from sac_gmm_agent import SAC_GMM_Agent
sys.path.insert(0, str(Path(__file__).parents[1]))
from peg.panda_peg_env import panda_peg_v2
from GMM.gmm import GMM

def main():
    cfg_env = {"show_gui": False, "with_force": False, "with_joint": False, "relative": True, 
        "with_noise": False, "reward_type": "shaped", "dt": 0.005}
    cfg_eval ={"max_steps": 500, "num_episodes": 5, "render": False}   

    gmm_model_names = [ "models/GMM/gmm_peg_v2_pose_5.mat",
                        "models/GMM/gmm_peg_v2_pose_6.mat",
                        "models/GMM/gmm_peg_v2_pose_7.mat",
                        "models/GMM/gmm_peg_v2_pose_8.mat",
                        "models/GMM/gmm_peg_v2_pose_9.mat",
                        "models/GMM/gmm_peg_v2_pose_10.mat",
                        "models/GMM/gmm_peg_v2_pose_11.mat",
                        "models/GMM/gmm_peg_v2_pose_12.mat",
                        "models/GMM/gmm_peg_v2_pose_13.mat",
                        "models/GMM/gmm_peg_v2_pose_14.mat",]

    sac_model_names = ["sac_gmm_noise_force_peg_v2_pose_5_80.pth",
                       "sac_gmm_noise_force_peg_v2_pose_6_20.pth",
                       "sac_gmm_noise_force_peg_v2_pose_7_30.pth",
                       "sac_gmm_noise_force_peg_v2_pose_8_20.pth",
                       "sac_gmm_noise_force_peg_v2_pose_9_220.pth",
                       "sac_gmm_noise_force_peg_v2_pose_10_20.pth",
                       "sac_gmm_noise_force_peg_v2_pose_11_160.pth",
                       "sac_gmm_noise_force_peg_v2_pose_12_10.pth",
                       "sac_gmm_noise_force_peg_v2_pose_13_10.pth",
                       "sac_gmm_noise_force_peg_v2_pose_14_20.pth",]

    for gmm_model_name, sac_model_name in zip(gmm_model_names, sac_model_names):
        gmm_model = GMM(gmm_model_name)
        env = panda_peg_v2(**cfg_env)
        agent = SAC_GMM_Agent(env, gmm_model)
        agent.load(sac_model_name)
        accuracy, mean_return, mean_length = agent.evaluate(**cfg_eval)
        agent.env.close()
        print(sac_model_name)
        print("Accuracy:", accuracy, "Mean return:", mean_return, "Mean length:", mean_length)

if __name__ == "__main__":
    main()