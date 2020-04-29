"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from RL_brain import QLearningTable

import gym
import math
import numpy as np
import matplotlib.pyplot as plt

TOTAL_EPISODES = 250
# TOTAL_STEPS = 250
SHOW_ENV = True
total_reward_list = [] 

env = gym.make('CartPole-v0')

##定義feature的bucket個數：buckets for continuous state values to be assigned to
# Observation space: [position of cart, velocity of cart, angle of pole, rotation rate of pole]
# Setting bucket size to 1 = ignoring the particular observation state 
n_buckets = (1, 5, 6, 3)                          

##State範圍(state bounds)
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bounds[1] = [-0.5, 0.5]
state_bounds[3] = [-math.radians(50), math.radians(50)]

#State離散化
def Discretization_state(observation, n_buckets, state_bounds):

    state = [0] * len(observation) 

    for i, s in enumerate(observation):

        #每個feature有不同的分配
        #lower- and upper-bounds for each feature in observation
        l, u = state_bounds[i][0], state_bounds[i][1] 
        
        if s <= l:      #低於下限,分配為0        
            state[i] = 0
        elif s >= u:    #高於上限,分配為最大值
            state[i] = n_buckets[i] - 1
        else:           #範圍內,依比例分配
            state[i] = int(((s - l) / (u - l)) * n_buckets[i])

    return tuple(state)

#學習過程相關參數：Learning related constants; factors determined by trial-and-error
get_epsilon = lambda i: max(0.01, min(1, 1.0 - math.log10((i+1)/25)))   # 貪婪度 epsilon-greedy, factor to explore randomly; discounted over time
get_lr = lambda i: max(0.01, min(0.5, 1.0 - math.log10((i+1)/25)))      # 學習率 learning rate; discounted over time lr = min(1,0.05+0.05*eps_cnt)   
#gamma = 0.99                                                           # 獎勵衰減值 reward discount factor

def update():

    print("env.action_space: ", env.action_space)
    print("env.action_space.n: ", env.action_space.n) #action的數量
    print("env.observation_space: ", env.observation_space)    
    print("env.observation_space.shape: ", env.observation_space.shape[0])
    print("env.observation_space.high: ", env.observation_space.high)
    print("env.observation_space.low: ", env.observation_space.low)
    
    for episode in range(TOTAL_EPISODES):

        #初始化 initial observation
        observation = env.reset()
        observation = Discretization_state(observation, n_buckets, state_bounds) #將feature連續值離散化
        total_reward = 0

        #隨時間變化,設定學習參數(每個回合更改一次,若每個action改一次效果不佳)
        epsilon = get_epsilon(episode)
        lr = get_lr(episode)
        RL.set_e_greedy(epsilon)
        RL.set_learning_rate(lr)

        while True:
  
            #更新環境 fresh env
            if(SHOW_ENV == True):
                env.render()

            #根據觀察環境選擇action(RL choose action based on observation)
            action = RL.choose_action(str(observation))

            #環境反饋reward(RL take action and get next observation and reward)
            observation_, reward, done, info = env.step(action)
            observation_ = Discretization_state(observation_, n_buckets, state_bounds)
         
            total_reward += reward

            #學習(RL learn from this transition)
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                total_reward_list.append(total_reward)
                print("Episode %s finish, epsilon= %.2f, lr=%.2f, reward:%s"%(episode, epsilon, lr, total_reward))
                break
       
    # end of game
    print('game over')    
    env.close()

    #Plot Total Reward of Every Episode
    plt.plot(np.arange(len(total_reward_list)),total_reward_list)
    plt.ylim(0,TOTAL_EPISODES + 10)
    plt.title("Training Reward Results")
    plt.ylabel("reward")
    plt.xlabel("# of episode")
    plt.show()

if __name__ == "__main__":
    RL = QLearningTable(actions=list(range(env.action_space.n)))
    update()
