# from maze_env import Maze
from the_cliff_maze_env_up import Maze
from RL_brain import QLearningTable

import numpy as np
import matplotlib.pyplot as plt

total_reward_list = []

TOTAL_EPISODES = 100
DISPLAY_ENV = True
SAVE_RESULT_ENABLED = False

def update():

    #學習TOTAL_EPISODES個回合
    for episode in range(TOTAL_EPISODES):

        #初始化觀測值
        observation = env.reset()
        total_reward = 0
        total_steps = 0

        while True:

            if(DISPLAY_ENV == True):
                env.render()

            total_steps += 1

            #RL大腦根據state的觀測值挑選action
            action = RL.choose_action(str(observation))

            #探索者執行此action,並獲得環境回饋的下一個state的觀測值,reward,done
            observation_, reward, done = env.step(action)
            total_reward += reward

            #RL從此序列中學習(state, action, reward, state_)
            RL.learn(str(observation), action, reward, str(observation_))

            #將下一個state的值傳到下一次循環(探索者移到下一個state)
            observation = observation_

            #若掉到黑格或黃圈,此回合結束
            if done:
                print("Episode #{} finished after {} timesteps, total rewards = {}".format(episode, total_steps, total_reward))
                total_reward_list.append(total_reward)
                break

    print("game over!")
    env.destroy()

    plt.plot(np.arange(0, TOTAL_EPISODES), total_reward_list)
    plt.xlabel("episodes")
    plt.ylabel("total_rewards")
    plt.title("Learning performance")

    if(SAVE_RESULT_ENABLED == True):
        plt.savefig("filename{}.png".format(TOTAL_EPISODES), format="png")
    else:
        plt.show()
    
if __name__=="__main__":
    
    env = Maze()
    RL = QLearningTable(actions = list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()