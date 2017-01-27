#randomly guess a set of parameters and over 1000 training 
#episodes get the best one

def get_episode_reward(env, observation, params):
	#this variable scores timesteps, each timestep gives 1 score
	t = 0
	#episode reward
	episode_reward = 0
	#loop to keep taking actions (till it can)
	while(t < 1000):
		#take dot of two vectors to move left/right
		if (np.inner(observation, params) < 0):
			action = 0
		else:
			action = 1
	#perform the action and get report
		observation, reward, done, info = env.step(action)
		#if we are out
		if done:
			print("Episode %d finished after %d timesteps, reward = %d"%(episode, t + 1, episode_reward + 1))
			break
		episode_reward += reward
		t += 1
	return episode_reward



#import the environment
import gym
from gym import wrappers
#math library of python
import numpy as np 

#create instance of Cartpole game
env = gym.make('CartPole-v0')

#save videos in this folder
env = wrappers.Monitor(env, 'cartpole-upload', force=True)

#randomly assign parameters
parameters = 2 * np.random.rand(4) - 1

#this will store the best parameter learnt so far
best_learnt_paramaters = parameters

#streak of >150 score
streak = 0

#longest streak
longest_streak = 0

#best score till now
best_reward = 0

#loop to run 100 episodes to train
#each episode uses the best set of parameters  
#discovered by randomness till now
for episode in range(1000):

	parameters = 2 * np.random.rand(4) - 1

	observation = env.reset()

	#run our episode with the random set of parameters
	reward = get_episode_reward(env, observation, parameters)

	#update best_learnt_parameters
	if(reward >= best_reward):
		best_reward = reward
		best_learnt_parameters = parameters


#test the BEST set of parameters to find out performance on 100 sets
for episode in range(200):

	observation = env.reset()

	#run our episode with the random set of parameters
	reward = get_episode_reward(env, observation, best_learnt_parameters)

	#update the streak and longest streak
	if(reward > 150):
		streak = streak + 1
		if(streak>longest_streak):
			longest_streak = streak
	else:
		streak = 0

print ("Longest streak >150 was %d"%(longest_streak))
env.monitor.close()