import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from panda_peg_env import  panda_peg_v2
from GMM.gmm import GMM

if __name__ == "__main__":
    # Environment hyperparameters
    env_params = {"show_gui": True, "with_force": False, "with_joint": False,
                  "relative": True, "with_noise": False, "dt": 0.01}
    env = panda_peg_v2(**env_params)

    # Evaluation parameters
    model_names = [ "gmm_peg_v2_pose_5.mat"]

    for model_name in model_names:
        model = GMM(model_name)
        eval_parameters = {"env":env, "num_episodes": 10, "max_steps": 500, 
                           "show_force":False, "render":False}
        accuracy, mean_return, mean_length = model.evaluate(**eval_parameters)
        print(model_name)
        print("Accuracy:", accuracy, "Mean return:", mean_return, "Mean length:", mean_length)
