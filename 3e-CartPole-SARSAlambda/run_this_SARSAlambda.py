"""
Sarsa is a online updating method for Reinforcement learning.

Unlike Q learning which is a offline updating method, Sarsa is updating while in the current trajectory.

You will see the sarsa is more coward when punishment is close because it cares about all behaviours,
while q learning is more brave because it only cares about maximum behaviour.
"""

from RL_brain import SarsaLambdaTable

import gym
import math
import numpy as np
import matplotlib.pyplot as plt

TOTAL_EPISODES = 250
DISPLAY_ENV = True
SHOW_ENV = True
SAVE_RESULT_ENABLED = False
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
#gamma = 0.99         

def update():

    #學習TOTAL_EPISODES個回合
    for episode in range(TOTAL_EPISODES):

        #初始化觀測值 initial observation
        observation = env.reset()
        observation = Discretization_state(observation, n_buckets, state_bounds) #將feature連續值離散化

        #SARSA根據state觀測選擇行為(RL choose action based on observation)
        action = RL.choose_action(str(observation))

        # total reward for this episode
        total_reward = 0
        time_steps = 0

        #隨時間變化,設定學習參數(每個回合更改一次,若每個action改一次效果不佳)
        epsilon = get_epsilon(episode)
        lr = get_lr(episode)
        RL.set_e_greedy(epsilon)
        RL.set_learning_rate(lr)

        while True:
            #更新環境 fresh env
            if(SHOW_ENV == True):
                env.render()

            time_steps += 1

            #在環境中採取行為,並得到下一個state_(observation_),reward和是否終止
            #RL take action and get next observation and reward
            observation_, reward, done, info = env.step(action)
            observation_ = Discretization_state(observation_, n_buckets, state_bounds)

            #根據下一個state_(observation_)選擇下一個action_
            #RL choose action based on next observation
            action_ = RL.choose_action(str(observation_))

            #從(s, a, r, s, a)中學習,更新QTable的參數
            #RL learn from this transition (s, a, r, s, a) ==> Sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)
            total_reward += reward

            #將下一個當成下一步的state(observation)和action
            #swap observation and action
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done:                
                print("Episode %s finish, %s timesteps, epsilon= %.2f, lr=%.2f, reward:%s"%(episode, time_steps, epsilon, lr, total_reward))
                total_reward_list.append(total_reward)
                break

    # end of game + plot learning performance
    print('game over')
    env.close()
    
    plt.plot(np.arange(0,TOTAL_EPISODES), total_reward_list)
    plt.xlabel("episodes")
    plt.ylabel("total_rewards")
    #plt.xlim()
    # plt.ylim(0, TOTAL_EPISODES + 10)
    plt.title("Learning performance")

    if(SAVE_RESULT_ENABLED == True):
        plt.savefig("filename{}.png".format(TOTAL_EPISODES), format="png")
    else:
        plt.show()
    

if __name__ == "__main__":
    RL = SarsaLambdaTable(actions=list(range(env.action_space.n)))
    update()
