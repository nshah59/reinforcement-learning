import numpy as np
import gym
import random
import time
from IPython.display import clear_output

# Gym - allows you to create environment in the game "Frozen Lake"
env = gym.make("FrozenLake-v0")


action_space_size = env.action_space.n
state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size,action_space_size))
print(q_table)

# Variables we will need for Q-Learning
num_episodes = 10000
# if by the 100th step the agent hasn't fallen in a hole or reached the end
# the episode will terminate
max_steps_per_episodes = 100

# Reminder - learning rate is the amount by which the agent updates q-values by
# new rewards; a 0.1 learning rate means that 0.1 of new q-values will be included as a
# weighted average for that state-action pair - denoted as ALPHA
learning_rate = 0.1

# Reminder - this is how much we discount future expected rewards by in our calculation
# of expected return - denoted as GAMMA
discount_rate = 0.99

# Part of exploration/exploitation tradeoff and the "epsilon-greedy" policy
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

#See how game scores change over time with:
rewards_all_episodes = []

# Complete Q learning and Training
# First for loop - everything that happens each episode
for episode in range(num_episodes):
    # To reset the environment back to the starting state
    state = env.reset()
    # Checks whether an episode is finished - starts False
    done = False
    #rewards at each episode
    rewards_current_episode = 0

    #second for loop - everything that happens at each step in each episode
    for step in range(max_steps_per_episodes):
        # Exploration-exploitation tradeoff
        # Determines Epsilon in some range
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            # Exploit the environment
            # Choose the action that has the highest q-value in the q-table for the
            # current state
            action = np.argmax(q_table[state, :])
        else:
            # Explore the environment
            # Sample an action randomly
            action = env.action_space.sample()

        # Call the action
        new_state, reward, done, info = env.step(action)

        # Update the q-table
        # From the formula for updating q-table
        q_table[state,action] = q_table[state,action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * np.max(q_table[new_state,:]))

        # Go to new state
        state = new_state
        # Add reward to list of rewards to keep tab
        rewards_current_episode += reward

        # If action in state ended state, break for loop
        if done == True:
            break

        # Exploration rate decay
        # Formula for exploration rate decay
    exploration_rate = min_exploration_rate + \
                        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
    rewards_all_episodes.append(rewards_current_episode)

# Now all episodes are done
# Use this to print average rewards per thousand episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
count = 1000
print("***********Average reward per thousand episodes ***********\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r / 1000)))
    count += 1000
print(q_table)


# For seeing the agent play in real life
for episode in range(3):
    state = env.reset()
    done = False
    print("EPISODE: ", episode + 1)
    time.sleep(1)

    for step in range(max_steps_per_episodes):
        clear_output(wait=True)
        env.render()
        time.sleep(0.3)

        action = np.argmax(q_table[state,:])
        new_state, reward, done, info = env.step(action)

        if done:
            clear_output(wait=True)
            env.render()
            if reward == 1:
                print("You reached the goal")
                time.sleep(3)
            else:
                print("You fell through a hole")
                time.sleep(3)
            clear_output(wait=True)
            break
        state = new_state
env.close()