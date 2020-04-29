"""
Agent learns the policy based on Q-learning with Q table.
Based on the example here: https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947
"""
import gym
import math
import numpy as np
import matplotlib.pyplot as plt

TOTAL_EPISODES = 250
# TOTAL_STEPS = 250
SHOW_ENV = True
total_reward_list = [] 

def choose_action(state, q_table, action_space, epsilon):
    if np.random.random_sample() < epsilon: # random action
        return action_space.sample() 
    else: # greddy action based on Q table
        return np.argmax(q_table[state]) 

def get_state(observation, n_buckets, state_bounds):
    state = [0] * len(observation) 
    for i, s in enumerate(observation):
        l, u = state_bounds[i][0], state_bounds[i][1] # lower- and upper-bounds for each feature in observation
        if s <= l:
            state[i] = 0
        elif s >= u:
            state[i] = n_buckets[i] - 1
        else:
            state[i] = int(((s - l) / (u - l)) * n_buckets[i])

    return tuple(state)

if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    # Preparing Q table
    ## buckets for continuous state values to be assigned to
    n_buckets = (1, 1, 6, 3) # Observation space: [position of cart, velocity of cart, angle of pole, rotation rate of pole]
                             # Setting bucket size to 1 = ignoring the particular observation state 

    ## discrete actions
    n_actions = env.action_space.n

    ## state bounds
    state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
    state_bounds[1] = [-0.5, 0.5]
    state_bounds[3] = [-math.radians(50), math.radians(50)]
    
    ## init Q table for each state-action pair
    q_table = np.zeros(n_buckets + (n_actions,))
    eligibility_trace = q_table.copy()

    # Learning related constants; factors determined by trial-and-error
    get_epsilon = lambda i: max(0.01, min(1, 1.0 - math.log10((i+1)/25))) # epsilon-greedy, factor to explore randomly; discounted over time
    get_lr = lambda i: max(0.01, min(0.5, 1.0 - math.log10((i+1)/25))) # learning rate; discounted over time
    gamma = 0.99 # reward discount factor
    lambda_ = 0.9

    # SARSA(lambda)
    for i_episode in range(TOTAL_EPISODES):

        #隨時間變化,設定學習參數(每個回合更改一次,若每個action改一次效果不佳)
        epsilon = get_epsilon(i_episode)
        lr = get_lr(i_episode)

        #初始化觀測值 initial observation
        observation = env.reset() # reset environment to initial state for each episode
        total_reward = 0 # accumulate rewards for each episode
        state = get_state(observation, n_buckets, state_bounds) # turn observation into discrete state
        time_steps = 0

        #SARSA根據state觀測選擇行為(RL choose action based on observation)
        action = choose_action(state, q_table, env.action_space, epsilon)#(str(observation))

        # for t in range(250):
        while True:
            #更新環境 fresh env    
            if(SHOW_ENV == True):
                env.render()

            time_steps += 1

            #在環境中採取行為,並得到下一個state_(observation_),reward和是否終止
            #RL take action and get next observation and reward
            observation_, reward, done, info = env.step(action) # do the action, get the reward
            total_reward += reward
            next_state = get_state(observation_, n_buckets, state_bounds)

            #根據下一個state_(observation_)選擇下一個action_
            #RL choose action based on next observation
            action_ = choose_action(next_state, q_table, env.action_space, epsilon)#(str(observation_))


            # 學習 learn
            q_predict = q_table[state + (action,)]
            q_target = reward + gamma * q_table[next_state+ (action_,)]
            error = q_target - q_predict

            eligibility_trace[state ] *= 0     #eligibility_trace.loc[state, :] *= 0
            eligibility_trace[state + (action,)] = 1 #eligibility_trace.loc[state, action] = 1

            #更新QTable
            q_table += lr * error* eligibility_trace

            #隨著時間衰減eligibility trace的值,離獲取reward越遠的步,他的"不可或缺性"越小
            # decay eligibility trace after update
            eligibility_trace *= gamma*lambda_

            # Transition to next state
            state = next_state
            action = action_

            if done:
                # print('Episode finished after {} timesteps, total rewards {}'.format(t+1, total_reward))
                total_reward_list.append(total_reward)
                print("Episode %s finish, steps = %s, epsilon= %.2f, lr=%.2f, reward:%s"%(i_episode, time_steps, epsilon, lr, total_reward))           
                break
    
    # end of game
    print('game over') 
    env.close() # need to close, or errors will be reported

    #Plot Total Reward of Every Episode
    plt.plot(np.arange(len(total_reward_list)),total_reward_list)
    plt.ylim(0,TOTAL_EPISODES + 10)
    plt.title("Training Reward Results")
    plt.ylabel("reward")
    plt.xlabel("# of episode")
    plt.show()