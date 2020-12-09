import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from panda_peg_env import  pandaPegV2
from GMM.gmm import GMM

if __name__ == "__main__":
    # Environment hyperparameters
    env_params = {"show_gui": False, "with_force": False, "with_joint": False,
                  "relative": True, "with_noise": True, "dt": 0.05}
    env = pandaPegV2(**env_params)

    # Evaluation parameters
    #model_names = [ "outputs/2020-12-02/08-42-22/models/GMM_models/peg_250.pth.npy"]
    #model_names = [ "models/GMM_models/gmm_peg_v2_13.mat"]
    #model_names = [ "models/optimizer/gmm_peg_v2_pose_9.npy"]
    #model_names = ["outputs/2020-12-08/14-12-01/models/optimizer/optimized.npy"]
    model_names = ["optimized.npy"]
    for model_name in model_names:
        model = GMM(model_name)
        eval_parameters = {"env":env, "num_episodes": 100, "max_steps": 500, 
                           "show_force":False, "render":False}
        accuracy, mean_return, mean_length = model.evaluate(**eval_parameters)
        print("Accuracy:", accuracy, "Mean return:", mean_return, "Mean length:", mean_length)
