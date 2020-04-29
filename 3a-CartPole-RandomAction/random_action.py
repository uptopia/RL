"""
Agent taking random actions.
Based on the example on the official website.
#由KEY SECTION可見,agent是沒有學習行為,故整體reward不高.
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

TOTAL_EPISODES = 100
TOTAL_STEPS = 250 

if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    print("env.action_space: ", env.action_space)
    print("env.observation_space: ", env.observation_space)
    print("env.action_space.n: ", env.action_space.n)
    print("env.observation_space.shape: ", env.observation_space.shape[0])
    print("env.observation_space.high: ", env.observation_space.high)
    print("env.observation_space.low: ", env.observation_space.low)

    total_rewards_list = []

    for i_episode in range(TOTAL_EPISODES):

        observation = env.reset()   # reset environment to initial state for each episode
        total_rewards = 0           # accumulate rewards for each episode

        for t in range(TOTAL_STEPS):#設定每個episode最多跑TOTAL_STEPS個action
            env.render()

            #-----------------------KEY SETCTION----------------------------    
            action = env.action_space.sample()                  #無論環境如何隨機進行action(choose a random action)
            observation, reward, done, info = env.step(action)  #do the action, get the reward
            total_rewards += reward
            #-----------------------KEY SETCTION----------------------------
                
            if done:
                print('Episode #{} finished after {} timesteps, total rewards {}'.format(i_episode, t+1, total_rewards))
                total_rewards_list.append(total_rewards)
                break

    env.close() # need to close, or errors will be reported

    #print training results
    plt.plot(np.arange(0, TOTAL_EPISODES), total_rewards_list)
    plt.xlabel("episodes")
    plt.ylabel("total_rewards")
    plt.title("Learning performance")
    plt.show()