"""
Agent taking reasonable actions based on hand-made rules,
i.e. go to right if leaning towards left, vice versa.

Observation space: [position of cart, velocity of cart, angle of pole, rotation rate of pole]
Action space: [moving cart to left, moving cart to right]

Note that good agents learn the policies by themselves,
and don't need to know the meaning of the observations.
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

TOTAL_EPISODES = 100
TOTAL_STEPS = 250 

total_reward_list = []

#引進Policy:柱子左傾,小車往左移
def choose_action(observation):
    pos, v, ang, rot = observation
    return 0 if ang < 0 else 1 # a simple rule based only on angles

if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    for i_episode in range(TOTAL_EPISODES):
        
        observation = env.reset()   # reset environment to initial state for each episode
        total_rewards = 0           # accumulate rewards for each episode

        for t in range(TOTAL_STEPS):
            env.render()

            #-------------------KEY SECTION--------------------
            action = choose_action(observation)                 # choose an action based on hand-made rule 
            observation, reward, done, info = env.step(action)  # do the action, get the reward
            total_rewards += reward
            #-------------------KEY SECTION--------------------

            if done:
                print('Episode #{} finished after {} timesteps, total rewards {}'.format(i_episode, t+1, total_rewards))
                total_reward_list.append(total_rewards)
                break

    env.close() # need to close, or errors will be reported

    #plot training result
    plt.plot(np.arange(0, TOTAL_EPISODES), total_reward_list)
    plt.xlabel("episodes")
    plt.ylabel("total_rewards")
    plt.title("Learning performance")
    plt.show()