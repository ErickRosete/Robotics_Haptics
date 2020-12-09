
from numpy.core.fromnumeric import ravel
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from SAC_GMM.networks import SoftQNetwork, PolicyNetwork
from Soft_Actor_Critic.replay_buffer import ReplayBuffer
from GMM.gmm import GMM

class SAC_GMM_Agent:
    def __init__(self, env, model, window_size=32, batch_size=256, gamma=0.99, tau=0.005, 
        actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, hidden_dim=256, encode_dim=8):
        
        #GMM 
        self.initial_model = model      # Initial model provided
        self.model = GMM() 
        self.model.copy_model(self.initial_model)    # Model used for training
        self.window_size = window_size

        #Environment
        self.env = env

        # We will change only the mu with the higher weight
        gmm_dim = model.priors.size + model.mu.size
        state_dim = env.observation_space.shape[0]  
        action_dim = model.priors.size + model.mu.size

        # Action_space
        priors_high = np.ones(model.priors.size) * 0.1
        mu_high = np.ones(model.mu.size) * 0.001
        action_high = np.concatenate((priors_high, mu_high), axis=-1)
        action_low = - action_high

        #Log 
        self.logger = logging.getLogger(__name__)
        self.writer = SummaryWriter()

        #Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        #Entropy
        self.alpha = 1
        self.target_entropy = -np.prod(env.action_space.shape).item()  # heuristic value
        self.log_alpha = torch.zeros(1, requires_grad=True, device="cuda")
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        #Networks
        self.Q1 = SoftQNetwork(state_dim, gmm_dim, action_dim, encode_dim, hidden_dim).cuda()
        self.Q1_target = SoftQNetwork(state_dim, gmm_dim, action_dim, encode_dim, hidden_dim).cuda()
        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q1_optimizer = optim.Adam(self.Q1.parameters(), lr=critic_lr)

        self.Q2 = SoftQNetwork(state_dim, gmm_dim, action_dim, encode_dim, hidden_dim).cuda()
        self.Q2_target = SoftQNetwork(state_dim, gmm_dim, action_dim, encode_dim, hidden_dim).cuda()
        self.Q2_target.load_state_dict(self.Q2.state_dict())
        self.Q2_optimizer = optim.Adam(self.Q2.parameters(), lr=critic_lr)

        self.actor = PolicyNetwork(state_dim, gmm_dim, action_dim, encode_dim,
                                    action_high, action_low, hidden_dim).cuda()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.loss_function = torch.nn.MSELoss()
        self.replay_buffer = ReplayBuffer()

    def getAction(self, state, deterministic=False):
        """Interface to get action from SAC Actor, ready to be used in the environment"""
        tensor_state = {k: torch.tensor(v, dtype=torch.float, device="cuda") for k, v in state.items()}
        action, _ = self.actor.getActions(**tensor_state, deterministic=deterministic, reparameterize=False) 
        return action.detach().cpu().numpy()

    def update(self, state, action, next_state, reward, done):

        self.replay_buffer.add_transition(state, action, next_state, reward, done)

        # Sample next batch and perform batch update: 
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = \
            self.replay_buffer.next_batch(self.batch_size, tensor=True)

        #Policy evaluation
        with torch.no_grad():
            policy_actions, log_pi = self.actor.getActions(**batch_next_states, deterministic=False, reparameterize=False)
            Q1_next_target = self.Q1_target(**batch_next_states, action=policy_actions)
            Q2_next_target = self.Q2_target(**batch_next_states, action=policy_actions)
            Q_next_target = torch.min(Q1_next_target, Q2_next_target)
            td_target = batch_rewards + (1 - batch_dones) * self.gamma * (Q_next_target - self.alpha * log_pi)

        # Critic update
        Q1_value = self.Q1(**batch_states, action=batch_actions)
        self.Q1_optimizer.zero_grad()
        Q1_loss = self.loss_function(Q1_value, td_target)
        Q1_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.Q1.parameters(), 1)
        self.Q1_optimizer.step()

        Q2_value = self.Q2(**batch_states, action=batch_actions)
        self.Q2_optimizer.zero_grad()
        Q2_loss = self.loss_function(Q2_value, td_target)
        Q2_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.Q2.parameters(), 1)
        self.Q2_optimizer.step()
        critic_loss = (Q1_loss.item() + Q2_loss.item())/2

        # Policy improvement
        policy_actions, log_pi = self.actor.getActions(**batch_states, deterministic=False, reparameterize=True)
        Q1_value = self.Q1(**batch_states, action=policy_actions)
        Q2_value = self.Q2(**batch_states, action=policy_actions)
        Q_value = torch.min(Q1_value, Q2_value)
        
        self.actor_optimizer.zero_grad()
        actor_loss = (self.alpha * log_pi - Q_value).mean()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optimizer.step()

        #Update entropy parameter 
        alpha_loss = (self.log_alpha * (-log_pi - self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
        
        #Update target networks
        self.soft_update(self.Q1_target, self.Q1, self.tau)
        self.soft_update(self.Q2_target, self.Q2, self.tau)
        
        return critic_loss, actor_loss.item(), alpha_loss.item()

    def evaluate(self, num_episodes = 5, max_steps = 2000, render=False):
        succesful_episodes, episodes_returns, episodes_lengths = 0, [], []
        for episode in range(1, num_episodes + 1):
            env_state = self.env.reset()
            self.model.copy_model(self.initial_model)
            sac_state = {"state": env_state, "gmm": self.model.get_state()}
            episode_return, episode_length, left_steps = 0, 0, max_steps
            while left_steps > 0:
                sac_action = self.getAction(sac_state, deterministic=True) 
                self.model.update_gaussians(sac_action)
                model_reward = 0
                for step in range(self.window_size): 
                    dim = self.model.mu.shape[0]//2
                    vel = self.model.predict_velocity(env_state[:dim])
                    env_next_state, reward, done, info = self.env.step(vel[:3])
                    model_reward += reward
                    env_state = env_next_state
                    episode_length += 1
                    left_steps -= 1
                    if render:
                        self.env.render()
                    if done or left_steps <= 0:
                        break
                episode_return += model_reward
                sac_state = {"state": env_state, "gmm": self.model.get_state()}
                if done:
                    break
            if ("success" in info) and info['success']:
                succesful_episodes += 1
            episodes_returns.append(episode_return) 
            episodes_lengths.append(episode_length)
        accuracy = succesful_episodes/num_episodes
        return accuracy, np.mean(episodes_returns), np.mean(episodes_lengths)

    def train(self, num_episodes, max_steps, log=True, eval_every=10, eval_episodes=5, 
        render=False, early_stopping=False, save_dir="models/", save_filename="sac_gmm_model", save_every=10): 
        
        episodes_returns, episodes_lengths = [], []
        for episode in range(1, num_episodes + 1):
            # Restart Environment and model
            env_state = self.env.reset()
            self.model.copy_model(self.initial_model)
            # Get SAC-state
            episode_return = 0 
            ep_steps, sac_steps = 0, 0
            ep_critic_loss, ep_actor_loss, ep_alpha_loss = 0, 0, 0
            left_steps = max_steps
            while left_steps > 0:
                sac_state = {"state": env_state, "gmm":self.model.get_state()}
                sac_action = self.getAction(sac_state)
                self.model.update_gaussians(sac_action)
                model_reward = 0
                for step in range(self.window_size): 
                    dim = self.model.mu.shape[0]//2
                    vel = self.model.predict_velocity(env_state[:dim])
                    env_next_state, reward, done, info = self.env.step(vel[:3])
                    model_reward += reward
                    env_state = env_next_state
                    ep_steps += 1
                    left_steps -= 1
                    if render:
                        self.env.render()
                    if done or left_steps <= 0:
                        break
                episode_return += model_reward
                sac_next_state = {"state": env_state, "gmm":self.model.get_state()}
                critic_loss, actor_loss, alpha_loss = self.update(sac_state, sac_action, sac_next_state, model_reward, done)
                ep_critic_loss += critic_loss
                ep_actor_loss += actor_loss
                ep_alpha_loss += alpha_loss
                sac_steps += 1
                sac_state = sac_next_state
                if done:
                    break

            # End of episode
            episodes_returns.append(episode_return)
            episodes_lengths.append(step)
            self.logger.info("Episode: %d   Return: %2f   Episode length: %d" % (episode, episode_return, ep_steps))
            if log:
                self.writer.add_scalar('Train/return', episode_return, episode)
                self.writer.add_scalar('Train/episode_length', ep_steps, episode)
                self.writer.add_scalar('Train/critic_loss', ep_critic_loss/sac_steps, episode)
                self.writer.add_scalar('Train/actor_loss', ep_actor_loss/sac_steps, episode)
                self.writer.add_scalar('Train/alpha_loss', ep_alpha_loss/sac_steps, episode)

            # Validation
            if episode % eval_every == 0 or episode == num_episodes:
                accuracy, eval_return, eval_length = self.evaluate(eval_episodes, max_steps)
                self.logger.info("Validation - Return: %2f   Episode length: %d" % (eval_return, eval_length))
                if log:
                    self.writer.add_scalar('Val/return', eval_return, episode)
                    self.writer.add_scalar('Val/episode_length', eval_length, episode)
                if accuracy == 1 and early_stopping:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)           
                    filename = "%s/%s_%d.pth"%(save_dir, save_filename, episode)
                    self.save(filename)
                    self.logger.info("Early stopped as accuracy in validation is 1.0")
                    break
                    
            # Save model
            if episode % save_every == 0 or episode == num_episodes:
                sac_dir = save_dir + "/SAC_GMM_models"
                gmm_dir = save_dir + "/GMM_models"
                if not os.path.exists(sac_dir):
                    os.makedirs(sac_dir)          
                if not os.path.exists(gmm_dir):
                    os.makedirs(gmm_dir) 
                filename = "%s/%s_%d.pth"%(sac_dir, save_filename, episode)
                self.save(filename)
                filename = "%s/%s_%d.npy"%(gmm_dir, save_filename, episode)
                self.model.save_model(filename)


        return episode, episodes_returns, episodes_lengths

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
    def save(self, file_name):
        torch.save({'actor_dict': self.actor.state_dict(),
                    'Q1_dict' : self.Q1.state_dict(),
                    'Q2_dict' : self.Q2.state_dict(),
                }, file_name)

    def load(self, file_name):
        if os.path.isfile(file_name):
            print("=> loading checkpoint... ")
            checkpoint = torch.load(file_name)
            self.actor.load_state_dict(checkpoint['actor_dict'])
            self.Q1.load_state_dict(checkpoint['Q1_dict'])
            self.Q2.load_state_dict(checkpoint['Q2_dict'])
            print("done !")
        else:
            print("no checkpoint found...")

    
