'''
- Q-Learning Off-Policy Control 
- Reinforcement Learning Tutorial


- This Python file contains driver code for the Q-Learning algorithm
- This Python file imports the class Q_Learning that implements the algorithm 
- The definition of the class Q_Learning is in the file "functions.py"

Name: Muhammad Bahriddinov
'''

# Note: 
# You can either use gym (not maintained anymore) or gymnasium (maintained version of gym)    
    
# tested on     
# gym==0.26.2
# gym-notices==0.0.8

# gymnasium==0.27.0
# gymnasium-notices==0.0.1

# Import necessary libraries
import gym
# instead of gym, you can use gymnasium 
# import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt 

# Import the Q_Learning class from the functions.py file
from functions import Q_Learning

# Create the CartPole environment
env = gym.make('CartPole-v1', render_mode='human')
# Reset the environment to get the initial state
(state, _) = env.reset()

# Define state discretization parameters
upperBounds = env.observation_space.high
lowerBounds = env.observation_space.low
cartVelocityMin = -3
cartVelocityMax = 3
poleAngleVelocityMin = -10
poleAngleVelocityMax = 10
upperBounds[1] = cartVelocityMax
upperBounds[3] = poleAngleVelocityMax
lowerBounds[1] = cartVelocityMin
lowerBounds[3] = poleAngleVelocityMin

numberOfBinsPosition = 30
numberOfBinsVelocity = 30
numberOfBinsAngle = 30
numberOfBinsAngleVelocity = 30
numberOfBins = [numberOfBinsPosition, numberOfBinsVelocity, numberOfBinsAngle, numberOfBinsAngleVelocity]

# Define Q-Learning algorithm parameters
alpha = 0.1       # Learning rate
gamma = 1         # Discount factor
epsilon = 0.2     # Exploration rate
numberEpisodes = 15000  # Number of episodes for training

# Create an instance of the Q_Learning class
Q1 = Q_Learning(env, alpha, gamma, epsilon, numberEpisodes, numberOfBins, lowerBounds, upperBounds)

# Run the Q-Learning algorithm to train the agent
Q1.simulateEpisodes()

# Simulate the learned strategy
(obtainedRewardsOptimal, env1) = Q1.simulateLearnedStrategy()

# Plot the sum of rewards per episode to visualize training progress
plt.figure(figsize=(12, 5))
plt.plot(Q1.sumRewardsEpisode, color='blue', linewidth=1)
plt.xlabel('Episode')
plt.ylabel('Sum of Rewards in Episode')
plt.yscale('log')  # Log scale for better visualization
plt.savefig('convergence.png')
plt.show()

# Close the environment after simulation
env1.close()

# Print the total rewards obtained by the optimal policy
print("Total rewards obtained by the optimal policy: ", np.sum(obtainedRewardsOptimal))

# Simulate a random strategy to compare with the optimal policy
(obtainedRewardsRandom, env2) = Q1.simulateRandomStrategy()

# Plot a histogram of rewards obtained by the random strategy
plt.hist(obtainedRewardsRandom)
plt.xlabel('Sum of rewards')
plt.ylabel('Percentage')
plt.savefig('histogram.png')
plt.show()

# Optionally, run the optimal strategy simulation again
(obtainedRewardsOptimal, env1) = Q1.simulateLearnedStrategy()
