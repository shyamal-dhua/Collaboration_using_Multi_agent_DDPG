# Collaboration_using_Multi_agent_DDPG

For this project, we will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

<img src="docs/description.gif">

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Observation vectors corresponding to 3 time instances are stacked together to give us a resultant observation of 24 variables. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, our agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Dependencies

To set up your python environment to run the code in this repository, follow the instructions below. Please note that we will be using `Anaconda`
for executing the code. 

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

2. Enter Anaconda terminal and create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	conda activate drlnd
	```
	  
3. Perform a minimal install of OpenAI gym.
  ```bash
  pip install gym
  ```

4. Clone this repository, and navigate to the `python/` folder.  Then, install several dependencies.
  ```bash
  cd Collaboration_using_Multi_agent_DDPG/python
  pip install .
  ```
5. Download whl file for `torch 0.4.0` from `http://download.pytorch.org/whl/cpu/torch-0.4.0-cp36-cp36m-win_amd64.whl`. Then install it using the command below. Also install `pywinpty` using `pipwin`, then install `jupyter` [Note: during installation of `jupyter` it tries to install a higher version of `pywinpty` and can give some error which we can ignore.]
  ```bash
  pip install --no-deps path_to_torch_whl_file\torch-0.4.0-cp36-cp36m-win_amd64.whl
  pip install pipwin
  pipwin install pywinpty
  pip install jupyter
  ```
6. Create an IPython kernel for the drlnd environment. We can use this kernel if we use `jupyter notebook` to run. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu.
  ```bash
  python -m ipykernel install --user --name drlnd --display-name "drlnd"
  ```

7. Extract the previously downloaded zip file for Environment [in step 1], to the root folder i.e. `Collaboration_using_Multi_agent_DDPG/`.

8. We need to run `main.py` for both training and testing. Follow the below instruction for training. We can set the total number of episodes, iterations per episode, batch size, starting value of epsilon, minimum value of epsilon and epsilon decay [for off-policy epsilon-greedy algorithm] during the training through command line. 
  ```bash
  python main.py --n_episodes 20000 --max_t 200 --batch_size 256 --eps_start 1 --eps_end 0.01 --eps_decay 0.995
  ```

9. Similarly follow the below instruction for testing. We need to provide a checkpoint file for both the actor and critic networks for this. 
  ```bash
  python main.py --evaluate_actor0 best_checkpoints/checkpoint_actor0_maddpg.pth --evaluate_critic0 best_checkpoints/checkpoint_critic0_maddpg.pth --evaluate_actor1 best_checkpoints/checkpoint_actor1_maddpg.pth --evaluate_critic1 best_checkpoints/checkpoint_critic1_maddpg.pth
  ```

10. To get better understanding on the available parameters to be passed in command line, use the help function as below, 
  ```bash
  python main.py -h
  ```
11. The model parameters get stored in the folder `Collaboration_using_Multi_agent_DDPG/checkpoints/`. The checkpoints get stored for any network scoring above +0.5. Our best trained checkpoints are kept in the folder `Collaboration_using_Multi_agent_DDPG/best_checkpoints/`. The plots for the training get stored in `Collaboration_using_Multi_agent_DDPG/plots/`.

12. The Jupyter notebook solution for MADDPG `collab_compet_maddpg.ipynb` and the project report `Report.ipynb` are present in the folder `Collaboration_using_Multi_agent_DDPG/report/`.

13. Below is how the agents were acting without training. The agents take random actions. 

https://user-images.githubusercontent.com/40301650/176344695-507171c7-6c4d-4f8a-9bd0-c19f290dd994.mp4

14. After training, the MADDPG solution looks as below. The agents successfully learn to keep the ball from falling.

https://user-images.githubusercontent.com/40301650/176345582-3bf7668d-ba32-411b-8cb2-106fe8efe795.mp4






