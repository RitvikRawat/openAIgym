import gym
import numpy as np
import random
import math
from gym import wrappers

def discretize(observation):

	if(observation[2] <= -0.26):
		i = 0
	elif(observation[2] <= -0.13):
		i = 1
	elif(observation[2] <= 0):
		i = 2
	elif(observation[2] <= 0.13):
		i = 3
	elif(observation[2] <= 0.26):
		i = 4
	else:
		i = 5

	if(observation[3] <= -0.29):
		j = 0
	elif(observation[3] <= 0.29):
		j = 1
	else:
		j = 2


	return (i,j)

# def discretize(observation):

# 	if(observation[0] <= -0.6):
# 		i = 0
# 	elif(observation[0] <= 0):
# 		i = 1
# 	else:
# 		i = 2

# 	if(observation[1] <= 0):
# 		j = 0
# 	else:
# 		j = 1

# 	return (i,j)

def get_action(state,explore_rate):

	if random.random() < explore_rate:
		action = env.action_space.sample()
	# Select the action with the highest q
	else:
		action = np.argmax(q_table[state])
	return action


env = gym.make('CartPole-v0')

env = wrappers.Monitor(env, 'CartPole-v0-QL', force=True)
#env = gym.wrappers.Monitor(env, 'CartPole-v0-QL', force=True)

q_table = np.zeros((6,3,2))


MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1
discount_factor = 0.99


explore_rate = 1
learning_rate = 1

num_streaks = 0
best_streak = 0

for episode in range(10000):

	obv = env.reset()

	i_state = discretize(obv)

	for t in range(1000):

		action = get_action(i_state, explore_rate)

		obv, reward, done, _ = env.step(action)

		state = discretize(obv)

		best_q = np.amax(q_table[state])
		q_table[i_state + (action,)] += learning_rate*(reward + discount_factor*(best_q) - q_table[i_state + (action,)])

		i_state = state

		# print("\nEpisode = %d" % episode)
		# print("t = %d" % t)
		# print("Action: %d" % action)
		# print("State: %s" % str(state))
		# print("Reward: %f" % reward)
		# print("Best Q: %f" % best_q)
		# print("Explore rate: %f" % explore_rate)
		# print("Learning rate: %f" % learning_rate)
		# print("Streaks: %d" % num_streaks)

		if done:
			print("Episode %d finished after %f time steps" % (episode, t))
			if(t >= 199):
				num_streaks += 1
				if(num_streaks>=best_streak):
					best_streak = num_streaks
			else:
				num_streaks = 0
			break
	# print("\nEpisode = %d" % episode)
	# print("t = %d" % t)
	# print("Action: %d" % action)
	# print("State: %s" % str(state))
	# print("Reward: %f" % reward)
	# print("Best Q: %f" % best_q)
	# print("Explore rate: %f" % explore_rate)
	# print("Learning rate: %f" % learning_rate)
	# print("Streaks: %d" % num_streaks)


	if num_streaks > 110:
		print("Hurrah we made it !")
		break

	# Update parameters
	explore_rate = max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((episode+1.0)/25.0)))
	learning_rate = max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((episode+1.0)/25.0)))
print("best streak is %d" % (best_streak))
env.monitor.close()