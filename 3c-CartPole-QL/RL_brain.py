"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd

class RL(object):
    #初始化
    def __init__(self, actions, learning_rate=0.05, reward_decay=0.9, e_greedy=0.3):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    #設定epsilon值
    def set_e_greedy(self, epsilon):
        self.epsilon = epsilon

    #設定學習率
    def set_learning_rate(self, lr):
        self.lr = lr

    #檢查狀態已存在
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    #根據觀測狀態選擇行為
    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() > self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    #每種演算法學習方式不同,須重新定義
    def learn(self, *arg):
        pass

class QLearningTable(RL):
    #初始化
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    #學習
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update


#on-policy
#SARSA(0)單步更新:每走完一步,更新一次行為準則
class SarsaTable(RL):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            #SARSA的q_target基於選好的a_,而不是Q(s_)的最大值
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal

        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  #更新QTable(update)

#後向觀測算法backward eligibility traces
#SARSA(lambda)回合更新:走完lambda步,等待回合完畢,再一次性更新
class SarsaLambdaTable(RL):

    #初始化
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9):
        super(SarsaLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

        # backward view, eligibility trace.
        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy() #只是紀錄每一回合的每一步,新回合開始時要清空為0

    #檢查state是否存在
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            to_be_append = pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            self.q_table = self.q_table.append(to_be_append)

            # also update eligibility trace
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    #學習更新參數
    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        error = q_target - q_predict

        # increase trace amount for visited state-action pair

        # Method 1:
        # 對於經歷過的state-action,我們讓它+1,證明他是得到reward路途中不可或缺的一環
        # self.eligibility_trace.loc[s, a] += 1

        # Method 2:
        # 更有效的方法
        self.eligibility_trace.loc[s, :] *= 0
        self.eligibility_trace.loc[s, a] = 1

        #更新QTable
        self.q_table += self.lr * error * self.eligibility_trace

        #隨著時間衰減eligibility trace的值,離獲取reward越遠的步,他的"不可或缺性"越小
        # decay eligibility trace after update
        self.eligibility_trace *= self.gamma*self.lambda_