# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
import numpy as np
from utilities import soft_update, transpose_to_tensor, transpose_list
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'



class MADDPG:
    def __init__(self, discount_factor=0.99, tau=1e-3):
        super(MADDPG, self).__init__()

        # critic input = obs_full + actions = 24+24+2+2=52
        self.maddpg_agent = [DDPGAgent(24, 400, 300, 2, 52, 256, 128), 
                             DDPGAgent(24, 400, 300, 2, 52, 256, 128)]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        #print(obs_all_agents[0].shape) # torch.Size([1, 24])
        #print(len(obs_all_agents)) # 2
        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        #print(len(actions)) # 2
        #print(actions[0].shape) # torch.Size([1, 2])
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        #print("len(obs_all_agents) = ", len(obs_all_agents)) # 2
        #print("obs_all_agents[0].shape = ", obs_all_agents[0].shape) # torch.Size([256, 24])
        target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        #print("len(target_actions) = ", len(target_actions)) # 2
        #print("target_actions[0].shape = ", target_actions[0].shape) # torch.Size([256, 2])
        return target_actions

    def update(self, samples, agent_number):
        """update the critics and actors of all the agents """
        
        obs, obs_full, action, reward, next_obs, next_obs_full, done = map(transpose_to_tensor, samples)
        #print("len(obs) = ", len(obs)) # 2
        #print("obs[0].shape = ", obs[0].shape) # torch.Size([256, 24])
        #print("len(obs_full) = ", len(obs_full)) # 48
        #print("obs_full[0].shape = ", obs_full[0].shape) # torch.Size([256])
        #print("len(reward) = ", len(reward)) # 2
        #print("reward[0].shape = ",reward[0].shape) # torch.Size([256])
        #print("len(action) = ", len(action)) # 2
        #print("action[0].shape = ", action[0].shape) # torch.Size([256, 2])

        obs_full = torch.stack(obs_full)
        next_obs_full = torch.stack(next_obs_full)
        #print("obs_full.shape = ", obs_full.shape) # torch.Size([48, 256])
        #print("next_obs_full.shape = ", next_obs_full.shape) # torch.Size([48, 256])
        
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_obs)
        target_actions = torch.cat(target_actions, dim=1)
        #print("target_actions.shape = ", target_actions.shape) # torch.Size([256, 4])
        
        target_critic_input = torch.cat((next_obs_full.t(),target_actions), dim=1).to(device)
        #print("target_critic_input.shape = ", target_critic_input.shape) # torch.Size([256, 52])
        
        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)
            #print("q_next.shape = ", q_next.shape) # torch.Size([256, 1])
        
        y = reward[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[agent_number].view(-1, 1))
        action = torch.cat(action, dim=1)
        critic_input = torch.cat((obs_full.t(), action), dim=1).to(device)
        q = agent.critic(critic_input)
        #print("critic_input.shape = ", critic_input.shape) # torch.Size([256, 52])
        #print("q.shape = ", q.shape) # torch.Size([256, 1])
        
        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1)
        agent.critic_optimizer.step()

        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [ self.maddpg_agent[i].actor(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(obs) ]
                
        q_input = torch.cat(q_input, dim=1)
        #print("q_input.shape = ", q_input.shape) # torch.Size([256, 4])
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        q_input2 = torch.cat((obs_full.t(), q_input), dim=1)
        #print("q_input2.shape = ", q_input2.shape) # torch.Size([256, 52])
        
        # get the policy gradient
        actor_loss = -agent.critic(q_input2).mean()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
            
            
            




