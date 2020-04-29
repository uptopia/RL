"""
A simple example for Reinforcement Learning using table lookup Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd
import time

np.random.seed(2)  # reproducible


N_STATES = 5                    # 狀態：the length of the 1 dimensional world
ACTIONS = ['left', 'right']     # 行為：available actions
EPSILON = 0.9                   # 貪婪度：greedy policy
ALPHA = 0.1                     # 學習率：learning rate
GAMMA = 0.9                     # 獎勵遞減值：discount factor
MAX_EPISODES = 15               # 最大回合數：maximum episodes
FRESH_TIME = 0.08               # 更新時間：fresh time for one move


#建立Q Table(紀錄state,action,reward)
def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))), # q_table initial values(全零初始)
        columns=actions,                    # actions's name
    )
    print("Initialize Q Table:\n")
    print(table,"\n")                          # show table
    return table

#定義如何選擇action：
#(1)EPSILON的機率，按照Q表最大值選擇行為
#(2)剩下的機率，隨機選擇行為
def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]  #選出此state的所有action值
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all())  # 非貪婪或此狀態未被探索過 act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:   # 貪婪模式 act greedy 
        action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name

#獎勵及與環境互動的方式
def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    if A == 'right':    # move right
        if S == N_STATES - 2:   # terminate
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:   # move left
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    return S_, R

#更新環境
def update_env(S, episode, step_counter, total_reward):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        print("\tEpisode %s: total_steps = %s; total_reward = %s" % (episode+1, step_counter, total_reward))
        time.sleep(0.5)
        
        # interaction = '\n' 
        # print('\r{}'.format(interaction), end='')
        # time.sleep(0.5)
        # print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

#強化學習
def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)  #初始 Q Table
    total_reward_list = []

    for episode in range(MAX_EPISODES):
        
        step_counter = 0
        S = 0                   #回合初始位置
        is_terminated = False   #回合是否結束
        total_reward = 0        #此回合總獎勵
        update_env(S, episode, step_counter, total_reward) #環境更新
        
        while not is_terminated:

            A = choose_action(S, q_table)   #選行為
            S_, R = get_env_feedback(S, A)  #執行行為,並從環境得到回饋(獎勵) take action & get next state and reward
            q_predict = q_table.loc[S, A]   #估算值(狀態行為的值) 
            total_reward += R

            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   #實際值(回合未結束) next state is not terminal
            else:
                q_target = R            #實際值(回合已結束) next state is terminal
                is_terminated = True    # terminate this episode

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  #更新QTable
            S = S_                      #探索者移動到下一個state(move to next state)

            update_env(S, episode, step_counter+1, total_reward)  #環境更新    
            step_counter += 1

    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
