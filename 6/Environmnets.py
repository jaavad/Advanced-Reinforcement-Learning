import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import copy




class BlockedMaze:
  
    actions = ['u', 'd', 'l', 'r']
    blocked = [(i,2) for i in range(8)]
    start_state = (3, 0)
    goal_state = (8, 5)
    x_lim = 8
    y_lim = 5
    step = 0
    
    def plot(self):
        plt.figure(figsize=(9, 6))
        plt.ylim([0,6])
        plt.xlim([0, 9])
        plt.text(self.start_state[0] + 0.5, self.start_state[1] + 0.5, "S")
        plt.text(self.goal_state[0] + 0.5, self.goal_state[1] + 0.5, "G")
        for coor in self.blocked:
            plt.text(coor[0] + 0.5, coor[1] + 0.5, "X", color='red', fontsize='24')
        plt.grid()
        
    def start(self):
        self.state = self.start_state
        return self.state
    
    def act(self, action):
        if self.step == 1000:
            self.blocked = self.blocked[1:]
            self.blocked.append((8, 2))
        self.step += 1
        if action == 'l':
            next_state = max(0, self.state[0] -1), self.state[1]
        elif action == 'r':
            next_state = min(self.x_lim, self.state[0] + 1), self.state[1]
        elif action == 'd':
            next_state = self.state[0], max(0, self.state[1] - 1)
        else:
            next_state = self.state[0], min(self.state[1] + 1, self.y_lim)
        if next_state in self.blocked:
            next_state = self.state
        self.state = next_state
        if self.state == self.goal_state:
            return self.state, 1, True
        else:
            return self.state, 0, False
    def reset(self):
        self.step = 0
        self.blocked = [(i,2) for i in range(8)]
        
    



class ShortcutMaze:
   
    actions = ['u', 'd', 'l', 'r']
    blocked = [(i,2) for i in range(1, 9)]
    start_state = (3, 0)
    goal_state = (8, 5)
    x_lim = 8
    y_lim = 5
    step = 0
    
    def plot(self):
        plt.figure(figsize=(9, 6))
        plt.ylim([0,6])
        plt.xlim([0, 9])
        plt.text(self.start_state[0] + 0.5, self.start_state[1] + 0.5, "S")
        plt.text(self.goal_state[0] + 0.5, self.goal_state[1] + 0.5, "G")
        for coor in self.blocked:
            plt.text(coor[0] + 0.5, coor[1] + 0.5, "X", color='red', fontsize='24')
        plt.grid()
        
    def start(self):
        self.state = self.start_state
        return self.state
    
    def act(self, action):
        if self.step == 3000:
            self.blocked = self.blocked[:-1]
        self.step += 1
        if action == 'l':
            next_state = max(0, self.state[0] -1), self.state[1]
        elif action == 'r':
            next_state = min(self.x_lim, self.state[0] + 1), self.state[1]
        elif action == 'd':
            next_state = self.state[0], max(0, self.state[1] - 1)
        else:
            next_state = self.state[0], min(self.state[1] + 1, self.y_lim)
        if next_state in self.blocked:
            next_state = self.state
        self.state = next_state
        if self.state == self.goal_state:
            return self.state, 1, True
        else:
            return self.state, 0, False
    def reset(self):
        self.step = 0
        self.blocked = [(i,2) for i in range(1, 9)]
        
