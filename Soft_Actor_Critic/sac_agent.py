from networks import SoftQNetwork, PolicyNetwork
from replay_buffer import ReplayBuffer
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import logging

class SAC_Agent:
    def __init__(self, env, batch_size=256, gamma=0.99, tau=0.005, 
        actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, hidden_dim=256):
        #Environment
        self.env = env
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]        
        action_high = self.env.action_space.high
        action_low = self.env.action_space.low

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
        self.Q1 = SoftQNetwork(state_dim, action_dim, hidden_dim).cuda()
        self.Q1_target = SoftQNetwork(state_dim, action_dim, hidden_dim).cuda()
        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q1_optimizer = optim.Adam(self.Q1.parameters(), lr=critic_lr)

        self.Q2 = SoftQNetwork(state_dim, action_dim, hidden_dim).cuda()
        self.Q2_target = SoftQNetwork(state_dim, action_dim, hidden_dim).cuda()
        self.Q2_target.load_state_dict(self.Q2.state_dict())
        self.Q2_optimizer = optim.Adam(self.Q2.parameters(), lr=critic_lr)

        self.actor = PolicyNetwork(state_dim, action_dim, action_high, action_low,
                                    hidden_dim).cuda()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.loss_function = torch.nn.MSELoss()
        self.replay_buffer = ReplayBuffer()

    def getAction(self, state, deterministic=False):
        """Interface to get action from SAC Actor, ready to be used in the environment"""
        state = torch.tensor(state, dtype=torch.float, device="cuda")
        action, _ = self.actor.getActions(state, deterministic, reparameterize=False) 
        return action.detach().cpu().numpy()

    def update(self, state, action, next_state, reward, done):

        self.replay_buffer.add_transition(state, action, next_state, reward, done)

        # Sample next batch and perform batch update: 
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = \
            self.replay_buffer.next_batch(self.batch_size, tensor=True)

        #Policy evaluation
        with torch.no_grad():
            policy_actions, log_pi = self.actor.getActions(batch_next_states, deterministic=False, reparameterize=False)
            Q1_next_target = self.Q1_target(batch_next_states, policy_actions)
            Q2_next_target = self.Q2_target(batch_next_states, policy_actions)
            Q_next_target = torch.min(Q1_next_target, Q2_next_target)
            td_target = batch_rewards + (1 - batch_dones) * self.gamma * (Q_next_target - self.alpha * log_pi)

        # Critic update
        Q1_value = self.Q1(batch_states, batch_actions)
        self.Q1_optimizer.zero_grad()
        Q1_loss = self.loss_function(Q1_value, td_target)
        Q1_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.Q1.parameters(), 1)
        self.Q1_optimizer.step()

        Q2_value = self.Q2(batch_states, batch_actions)
        self.Q2_optimizer.zero_grad()
        Q2_loss = self.loss_function(Q2_value, td_target)
        Q2_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.Q2.parameters(), 1)
        self.Q2_optimizer.step()
        critic_loss = (Q1_loss.item() + Q2_loss.item())/2

        # Policy improvement
        policy_actions, log_pi = self.actor.getActions(batch_states, deterministic=False, reparameterize=True)
        Q1_value = self.Q1(batch_states, policy_actions)
        Q2_value = self.Q2(batch_states, policy_actions)
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
            state = self.env.reset()
            episode_return = 0
            for step in range(max_steps):
                action = self.getAction(state, deterministic = True) 
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                episode_return += reward
                if render:
                    self.env.render()
                if done:
                    break
            if ("success" in info) and info['success']:
                succesful_episodes += 1
            episodes_returns.append(episode_return) 
            episodes_lengths.append(step)
        accuracy = succesful_episodes/num_episodes
        return accuracy, np.mean(episodes_returns), np.mean(episodes_lengths)

    def train(self, num_episodes, max_steps, exploration_episodes=0,
        log=True, eval_every=10, eval_episodes=5, render=False, early_stopping=False,
        save_dir="models/SAC_models", save_filename="sac_model", save_every=10): 

        episodes_returns, episodes_lengths = [], []
        for episode in range(1, num_episodes + 1):
            state = self.env.reset()
            episode_return = 0
            ep_critic_loss, ep_actor_loss, ep_alpha_loss = 0, 0, 0
            for step in range(max_steps):
                if episode < exploration_episodes:
                    action = self.env.action_space.sample()
                else:
                    action = self.getAction(state, deterministic = False) 
                next_state, reward, done, info = self.env.step(action)

                critic_loss, actor_loss, alpha_loss = self.update(state, action, next_state, reward, done)
                ep_critic_loss += critic_loss
                ep_actor_loss += actor_loss
                ep_alpha_loss += alpha_loss

                state = next_state
                episode_return += reward

                if render:
                    self.env.render()
                if done:
                    break

            # End of episode
            episodes_returns.append(episode_return)
            episodes_lengths.append(step)
            self.logger.info("Episode: %d   Return: %2f   Episode length: %d" % (episode, episode_return, step))
            if log:
                self.writer.add_scalar('Train/return', episode_return, episode)
                self.writer.add_scalar('Train/episode_length', step, episode)
                self.writer.add_scalar('Train/critic_loss', ep_critic_loss/step, episode)
                self.writer.add_scalar('Train/actor_loss', ep_actor_loss/step, episode)
                self.writer.add_scalar('Train/alpha_loss', ep_alpha_loss/step, episode)     

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
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)           
                filename = "%s/%s_%d.pth"%(save_dir, save_filename, episode)
                self.save(filename)

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

    
