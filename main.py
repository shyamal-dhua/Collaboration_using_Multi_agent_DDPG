#Import necessary packages
from unityagents import UnityEnvironment
import numpy as np
import random
import argparse
import json
import torch
from collections import deque
import matplotlib.pyplot as plt
from buffer import ReplayBuffer
from maddpg import MADDPG
from utilities import transpose_list, transpose_to_tensor

import warnings
warnings.filterwarnings('ignore')

def seeding(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)

"""Training code -> train()
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        batchsize (int): Number of samples to pick from replay buffer during network updates
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    
"""
def train(n_episodes=20000, max_t=200, batchsize = 256, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    # Load the environment.
    # Change the file_name parameter to match the location of the Unity environment that you downloaded.
    # Keeping 'no_graphics=True' helps reduce training time. Remove it to be able to visualise the environment while training.
    env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis.exe", no_graphics=True, seed=0)
    #env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis.exe", seed=0)
    print("\n")
    
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    # reset the environment for train
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    
    seeding(0)
    parallel_envs = 1
    update_count = 10
    buffer = ReplayBuffer(int(1e6)) # initialize a replay buffer
    maddpg = MADDPG() # initialize actors and critics
    
    scores_all = []                       # list containing scores from each episode
    scores_window = deque(maxlen=100)     # last 100 scores
    eps = eps_start                       # initialize epsilon
    max_mean = -np.inf
    i_episode_solved = 0
    first_solve = True
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        states = env_info.vector_observations # get the current state (for each agent)
        states_exp = np.expand_dims(states, axis = 0) # (1, 2, 24)
        scores_episode = np.zeros(num_agents) # initialize the score (for each agent)
        for t in range(max_t):
            # select an action (for each agent)
            actions = maddpg.act(transpose_to_tensor(states_exp), noise=eps) # transpose_to_tensor(states_exp)[0] -> torch.Size([1, 24])
            actions_array = torch.stack(actions).detach().numpy() # (2, 1, 2)
            actions_for_env = np.rollaxis(actions_array,1) # (1, 2, 2) 
            
            # send all actions to tne environment
            env_info = env.step(actions_for_env[0])[brain_name]
            
            next_states = env_info.vector_observations   # get next state (for each agent)
            next_states_exp = np.expand_dims(next_states, axis = 0) # (1, 2, 24)
            next_states_exp_full = np.reshape(next_states_exp, (1, num_agents*next_states_exp.shape[2])) # (1, 48)
            states_exp_full = np.reshape(states_exp, (1, num_agents*states_exp.shape[2])) # (1, 48)
            rewards = env_info.rewards                   # get reward (for each agent)
            dones = env_info.local_done                  # see if episode has finished (for any agent)
            rewards_exp = np.expand_dims(np.array(rewards), axis = 0) # (1, 2)
            dones_exp = np.expand_dims(np.array(dones), axis = 0) # (1, 2)
            
            # add data to buffer
            transition = (states_exp, states_exp_full, actions_for_env, rewards_exp, next_states_exp, next_states_exp_full, dones_exp)
            buffer.push(transition)
            
            states_exp = next_states_exp                 # roll over states to next time step
            scores_episode += rewards                    # update the score (for each agent)
            if np.any(dones):                            # exit loop if episode finished
                break
        
        # update the actor and critic networks for all agents
        if len(buffer) > batchsize:
            for i in range(update_count):
                for a_i in range(num_agents):
                    samples = buffer.sample(batchsize)
                    maddpg.update(samples, a_i)
                maddpg.update_targets()                  #soft update the target network towards the actual networks
        
        scores_window.append(np.max(scores_episode))
        scores_all.append(np.max(scores_episode))
        eps = max(eps_end, eps_decay*eps)                # decrease epsilon or noise
        print('\rEpisode {}\tAverage Score: {:.4f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.4f}'.format(i_episode, np.mean(scores_window)))
        
        #Store the weights corresponding to the best mean score
        curr_mean = np.mean(scores_window)
        if ((curr_mean >= 0.5) and (curr_mean > max_mean)):
            max_mean = curr_mean
            if (first_solve == True):
                first_solve = False
                i_episode_solved = i_episode
            #Save the best checkpoints
            torch.save(maddpg.maddpg_agent[0].actor.state_dict(), 'checkpoints/checkpoint_actor0_maddpg.pth')
            torch.save(maddpg.maddpg_agent[0].critic.state_dict(), 'checkpoints/checkpoint_critic0_maddpg.pth')
            torch.save(maddpg.maddpg_agent[1].actor.state_dict(), 'checkpoints/checkpoint_actor1_maddpg.pth')
            torch.save(maddpg.maddpg_agent[1].critic.state_dict(), 'checkpoints/checkpoint_critic1_maddpg.pth')
            #break
    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.4f}'.format(i_episode_solved - 100, max_mean))
    env.close()
    return scores_all
    
"""Testing code -> test()
    
    Params
    ======
        actor0_checkpoint_path: File path for the saved model checkpoints of actor 0 network
        critic0_checkpoint_path: File path for the saved model checkpoints of critic 0 network
        actor1_checkpoint_path: File path for the saved model checkpoints of actor 1 network
        critic1_checkpoint_path: File path for the saved model checkpoints of critic 1 network
    
"""
def test(actor0_checkpoint_path, critic0_checkpoint_path, actor1_checkpoint_path, critic1_checkpoint_path):
    # Load the environment
    # Change the file_name parameter to match the location of the Unity environment that you downloaded.
    env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis.exe")
    print("\n")
    
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    # reset the environment for test
    env_info = env.reset(train_mode=False)[brain_name]
    num_agents = len(env_info.agents)
    
    # load the weights from file
    maddpg = MADDPG() # initialize actors and critics
    maddpg.maddpg_agent[0].actor.load_state_dict(torch.load(actor0_checkpoint_path))
    maddpg.maddpg_agent[0].critic.load_state_dict(torch.load(critic0_checkpoint_path))
    maddpg.maddpg_agent[1].actor.load_state_dict(torch.load(actor1_checkpoint_path))
    maddpg.maddpg_agent[1].critic.load_state_dict(torch.load(critic1_checkpoint_path))
    
    #Run the evaluation
    states = env_info.vector_observations # get the current state (for each agent)
    states_exp = np.expand_dims(states, axis = 0)
    scores_episode = np.zeros(num_agents) # initialize the score (for each agent)

    for t in range(40000):
        actions = maddpg.act(transpose_to_tensor(states_exp), noise=0) # select an action (for each agent)
        actions_array = torch.stack(actions).detach().numpy()
        actions_for_env = np.rollaxis(actions_array,1)
        env_info = env.step(actions_for_env[0])[brain_name]            # send all actions to tne environment
        next_states = env_info.vector_observations                     # get next state (for each agent)
        next_states_exp = np.expand_dims(next_states, axis = 0)
        rewards = env_info.rewards                                     # get reward (for each agent)
        dones = env_info.local_done                                    # see if episode has finished (for any agent)
        states_exp = next_states_exp                                   # roll over states to next time step
        scores_episode += rewards                                      # update the score (for each agent)
        if np.any(dones):                                              # exit loop if episode finished
            break

    print('Total score (max of {} agents) in Test: {}'.format(num_agents, np.max(scores_episode)))
    print('Total score for agent 0 in Test: {}'.format(scores_episode[0]))
    print('Total score for agent 1 in Test: {}'.format(scores_episode[1]))
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch training script')
    
    parser.add_argument('--evaluate_actor0', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate for actor 0 (file name)')
    parser.add_argument('--evaluate_critic0', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate for critic 0 (file name)')
    parser.add_argument('--evaluate_actor1', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate for actor 1 (file name)')
    parser.add_argument('--evaluate_critic1', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate for critic 1 (file name)')
    parser.add_argument('--n_episodes', default=2000, type=int, metavar='N', help='maximum number of training episodes')
    parser.add_argument('--max_t', default=200, type=int, metavar='N', help='maximum number of timesteps per episode')
    parser.add_argument('--batch_size', default=256, type=int, metavar='N', help='number of samples from replay buffer')
    parser.add_argument('--eps_start', default=1.0, type=float, metavar='N', help='starting value of epsilon')
    parser.add_argument('--eps_end', default=0.01, type=float, metavar='N', help='minimum value of epsilon')
    parser.add_argument('--eps_decay', default=0.995, type=float, metavar='N', help='multiplicative factor for epsilon decay')
    
    args = parser.parse_args()
    
    if ((args.evaluate_actor0) and (args.evaluate_critic0) and (args.evaluate_actor1) and (args.evaluate_critic1)):
        print("\nRunning Test with the below parameters:\n")
        print("checkpoint path for actor 0 = ", args.evaluate_actor0)
        print("checkpoint path for critic 0 = ", args.evaluate_critic0)
        print("checkpoint path for actor 1 = ", args.evaluate_actor1)
        print("checkpoint path for critic 1 = ", args.evaluate_critic1)
        print("\n")
        test(args.evaluate_actor0, args.evaluate_critic0, args.evaluate_actor1, args.evaluate_critic1)
        exit(0)
    else:
        print("\nRunning Train with the below parameters:\n")
        print("n_episodes = ", args.n_episodes)
        print("max_t = ", args.max_t)
        print("batch_size = ", args.batch_size)
        print("eps_start = ", args.eps_start)
        print("eps_end = ", args.eps_end)
        print("eps_decay = ", args.eps_decay)
        print("\n")
        
        scores = train(args.n_episodes, args.max_t, args.batch_size, args.eps_start, args.eps_end, args.eps_decay)
        
        # save the scores
        with open("scores_json/maddpg_scores", "w") as fp:
            json.dump(scores, fp)
        # plot the scores
        # Plot the scores
        i = 0
        window_size = 100
        # Initialize an empty list to store moving averages
        moving_averages = []

        zeroes_list = [0] * (window_size-1)
        new_scores = zeroes_list + scores

        # Loop through the array t o
        #consider every window of size 3
        while i < len(new_scores) - window_size + 1:
          
            # Calculate the average of current window
            window_average = round(np.sum(new_scores[
              i:i+window_size]) / window_size, 4)
              
            # Store the average of current
            # window in moving average list
            moving_averages.append(window_average)
              
            # Shift window to right by one position
            i += 1

        fig = plt.figure()
        ax = fig.add_subplot(111)
        x_axis_values = np.arange(len(scores))
        plt.plot(x_axis_values, scores, label='scores')
        plt.plot(x_axis_values, moving_averages, label='moving average of 100')
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.legend()
        plt.show()
        fig.savefig("plots/training_plot_maddpg.pdf", dpi=fig.dpi)
        fig.savefig("plots/training_plot_maddpg.jpg", dpi=fig.dpi)
        exit(0)
    
    
    
    