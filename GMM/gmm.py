import os
import sys
import numpy as np   
import matlab.engine
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from utils.force_plot import ForcePlot
from utils.utils import get_cwd
from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, model_name=None):
        if model_name is not None:
            if not os.path.isfile(model_name):
                raise Exception("File not found")
            _, file_extension = os.path.splitext( model_name )
            if file_extension == ".npy":
                self.load_model(model_name)
            elif file_extension == ".mat":
                self.load_matlab_model(model_name)
            else:
                raise Exception("Extension not supported")

    def load_model(self, model_name):
        if Path(model_name).is_file():
            model = np.load(model_name, allow_pickle=True).item()
            self.priors = model['priors'].squeeze()
            self.mu = model['mu']
            self.sigma = model['sigma']
            print("File loaded succesfully")
        else:
            print("File doesn't exist")

    def load_matlab_model(self, model_name):
        if Path(model_name).is_file():
            eng = matlab.engine.start_matlab()
            eng.addpath(str(get_cwd() / "GMM"))
            priors, mu, sigma = eng.get_model(model_name, nargout=3)
            self.priors = np.asarray(priors).squeeze()
            self.mu = np.asarray(mu)
            self.sigma = np.asarray(sigma)
            eng.quit()
            print("File loaded succesfully")
        else:
            print("File doesn't exist")

    def copy_model(self, model):
        self.priors = np.copy(model.priors)
        self.mu = np.copy(model.mu)
        self.sigma = np.copy(model.sigma)

    def get_state(self):
        return np.concatenate((self.priors.ravel(), self.mu.ravel()), axis=-1)

    def get_main_gaussian(self, x):
        weights = self.get_weights(x)
        k = np.argmax(weights)
        return k, self.priors[k], self.mu[:, k], self.sigma[:,:,k]
    
    def update_gaussians(self, x):
        d_priors = x[:self.priors.size]
        self.priors += d_priors
        self.priors[self.priors < 0] = 0
        self.priors /= self.priors.sum()

        d_mu = x[self.priors.size:]
        d_mu = d_mu.reshape(self.mu.shape)
        self.mu += d_mu

    def update_main_gaussian(self, x, d_mu):
        k = np.argmax(self.get_weights(x))
        self.mu[:, k] += d_mu

    def get_weights(self, x):
        if x.ndim == 1:
            x = x.reshape(1,-1)
        
        batch_size = x.shape[0]
        dim = x.shape[1]
        num_gaussians = self.mu.shape[1]
        assert 2 * dim == self.mu.shape[0]

        weights = np.zeros((num_gaussians, batch_size))
        for i in range(num_gaussians):
            state_mu = self.mu[0:dim, i]
            state_sigma = self.sigma[0:dim, 0:dim, i]
            weights[i] = self.priors[i] * multivariate_normal.pdf(x, state_mu, state_sigma)
        weights /= (np.sum(weights, axis=0) + sys.float_info.epsilon) 
        return weights
    
    def predict_velocity(self, x):
        """ 
        Input
        x: np_array representing the current state relative to the target (Batch_size, state_dim) or (state_dim,)
        Output
        vel_mean: np_array represing the predicted velocity (Batch_size, State_dim) or (state_dim,)
        """
        if x.ndim == 1:
            x = x.reshape(1,-1)

        batch_size = x.shape[0]
        dim = x.shape[1]
        num_gaussians = self.mu.shape[1]
        assert 2 * dim == self.mu.shape[0]

        weights = self.get_weights(x)

        vel_mean = np.zeros((batch_size, dim))
        for i in range(num_gaussians):
            state_mu = self.mu[0:dim, i]
            vel_mu = self.mu[dim:2*dim, i]
            state_sigma = self.sigma[0:dim, 0:dim, i]
            cc_sigma = self.sigma[dim:2*dim, 0:dim, i]
            aux = vel_mu + (cc_sigma @ np.linalg.pinv(state_sigma) @ (x - state_mu).T).T # batch_size x dim
            vel_mean += weights[i].reshape(-1, 1) * aux
        return vel_mean.squeeze()

    def evaluate(self, env, max_steps=2000, num_episodes=10, show_force=False, render=False):
        succesful_episodes, episodes_returns, episodes_lengths = 0, [], []
        if show_force:
            plot = ForcePlot()
        for episode in range(1, num_episodes + 1):
            state = env.reset()
            episode_return = 0
            for step in range(max_steps):
                action = self.predict_velocity(state) 
                next_state, reward, done, info = env.step(action[:3])
                state = next_state
                episode_return += reward
                if render:
                    env.render()
                if done:
                    break
            if info['success']:
                succesful_episodes += 1
            episodes_returns.append(episode_return) 
            episodes_lengths.append(step)
        accuracy = succesful_episodes/num_episodes
        return accuracy, np.mean(episodes_returns), np.mean(episodes_lengths)

    def save_model(self, model_name):
        model = {
            "priors": self.priors,      # num_gaussians
            "mu": self.mu,              # observation_size * 2, num_gaussians
            "sigma": self.sigma         # observation_size * 2, observation_size * 2, num_gaussians
        }
        np.save(model_name, model)


def test():
    model = GMM("models/GMM_models/gmm_peg_v2_pose_9.npy")
    state = np.random.rand(16, 3)
    vel = model.predict_velocity(state)
    print(vel)
    state = np.random.rand(3)
    vel = model.predict_velocity(state)
    print(vel)

if __name__ == "__main__":
    test()