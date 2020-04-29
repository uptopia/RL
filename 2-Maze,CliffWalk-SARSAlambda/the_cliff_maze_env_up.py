# -*- coding: utf-8 -*-

"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""


import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


UNIT = 40   # pixels
MAZE_H = 4  # grid height
MAZE_W = 12  # grid width


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT)) #設定畫布大小
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',   #設定畫布大小（？
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

       # create origin
        origin = np.array([20, 20]) + np.array([0, UNIT * 3])


        # hell
        hell1_center = origin + np.array([UNIT , 0])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        # hell
        hell2_center = origin + np.array([UNIT * 2, 0])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')
        # hell
        hell3_center = origin + np.array([UNIT * 3, 0])
        self.hell3 = self.canvas.create_rectangle(
            hell3_center[0] - 15, hell3_center[1] - 15,
            hell3_center[0] + 15, hell3_center[1] + 15,
            fill='black')
        # hell
        hell4_center = origin + np.array([UNIT * 4, 0])
        self.hell4 = self.canvas.create_rectangle(
            hell4_center[0] - 15, hell4_center[1] - 15,
            hell4_center[0] + 15, hell4_center[1] + 15,
            fill='black')
        # hell
        hell5_center = origin + np.array([UNIT * 5, 0])
        self.hell5 = self.canvas.create_rectangle(
            hell5_center[0] - 15, hell5_center[1] - 15,
            hell5_center[0] + 15, hell5_center[1] + 15,
            fill='black')
        # hell
        hell6_center = origin + np.array([UNIT * 6, 0])
        self.hell6 = self.canvas.create_rectangle(
            hell6_center[0] - 15, hell6_center[1] - 15,
            hell6_center[0] + 15, hell6_center[1] + 15,
            fill='black')        
        # hell
        hell7_center = origin + np.array([UNIT * 7, 0])
        self.hell7 = self.canvas.create_rectangle(
            hell7_center[0] - 15, hell7_center[1] - 15,
            hell7_center[0] + 15, hell7_center[1] + 15,
            fill='black')
        # hell
        hell8_center = origin + np.array([UNIT * 8, 0])
        self.hell8 = self.canvas.create_rectangle(
            hell8_center[0] - 15, hell8_center[1] - 15,
            hell8_center[0] + 15, hell8_center[1] + 15,
            fill='black')
        # hell
        hell9_center = origin + np.array([UNIT * 9, 0])
        self.hell9 = self.canvas.create_rectangle(
            hell9_center[0] - 15, hell9_center[1] - 15,
            hell9_center[0] + 15, hell9_center[1] + 15,
            fill='black')
        # hell
        hell10_center = origin + np.array([UNIT * 10, 0])
        self.hell10 = self.canvas.create_rectangle(
            hell10_center[0] - 15, hell10_center[1] - 15,
            hell10_center[0] + 15, hell10_center[1] + 15,
            fill='black')
        # hell
        hell11_center = origin + np.array([UNIT * 0, UNIT * 1])
        self.hell11 = self.canvas.create_rectangle(
            hell11_center[0] - 15, hell11_center[1] - 15,
            hell11_center[0] + 15, hell11_center[1] + 15,
            fill='black')
         # hell
        hell12_center = origin + np.array([UNIT * 11, UNIT * 1])
        self.hell12 = self.canvas.create_rectangle(
            hell12_center[0] - 15, hell12_center[1] - 15,
            hell12_center[0] + 15, hell12_center[1] + 15,
            fill='black')            

        # create oval
        oval_center = origin + np.array([UNIT * 11, 0])
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.05)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20]) + np.array([0, UNIT * 3]) #記得修改起始位置
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # return observation
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state

        # reward function
        if s_ == self.canvas.coords(self.oval):
            reward = 500
            done = True
            s_ = 'terminal'
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2), self.canvas.coords(self.hell3),
        self.canvas.coords(self.hell4), self.canvas.coords(self.hell5), self.canvas.coords(self.hell6),
        self.canvas.coords(self.hell7), self.canvas.coords(self.hell8), self.canvas.coords(self.hell9),
        self.canvas.coords(self.hell10), self.canvas.coords(self.hell11), self.canvas.coords(self.hell12)]:
            reward = -100
            done = True
            s_ = 'terminal'
        else:
            reward = -1
            done = False

        return s_, reward, done

    def render(self):
        time.sleep(0.01)
        self.update()


