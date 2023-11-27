import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import copy

from Environmnets import BlockedMaze,   ShortcutMaze
from Algorithms import DynaQ, DynaQPlus, DynaQPlus_no_Footnote


def Blocked_Maze():
    
    blockedMaze = BlockedMaze()

    dynaQ_rewards = []
    dynaQplus_rewards = []
    for i in range(30):
        blockedMaze.reset()
        dynaQ = DynaQ(blockedMaze, 0.8, 0.1, 100, 0.95)
        dynaQ.train(200, max_timestep=3000)
        dynaQ_rewards.append(np.cumsum(dynaQ.rewards[:3000]))
    
    for i in range(30):
        blockedMaze.reset()
        dynaQplus = DynaQPlus(blockedMaze, 0.8, 0.1, 100, 1e-4, 0.95)
        dynaQplus.train(200, max_timestep=3000)
        dynaQplus_rewards.append(np.cumsum(dynaQplus.rewards[:3000]))
    
    plt.plot(range(3000), np.mean(dynaQ_rewards, axis=0), label='dynaQ')
    plt.plot(range(3000), np.mean(dynaQplus_rewards, axis=0), label='dynaQ+')
    plt.legend()
    plt.title('Figure 8.4')
    plt.xlabel('time steps')
    plt.ylabel('cummulative reward')
    plt.show()
    
    



def Blocked_Maze_no_footnote():
    
    blockedMaze = BlockedMaze()

    dynaQ_rewards = []
    dynaQplus_rewards = []
    dynaQplus_NoF_rewards = []
    
    for i in range(30):
        blockedMaze.reset()
        dynaQ = DynaQ(blockedMaze, 0.8, 0.1, 100, 0.95)
        dynaQ.train(200, max_timestep=3000)
        dynaQ_rewards.append(np.cumsum(dynaQ.rewards[:3000]))
    
    for i in range(30):
        blockedMaze.reset()
        dynaQplus_NF = DynaQPlus(blockedMaze, 0.8, 0.1, 100, 1e-4, 0.95)
        dynaQplus_NF.train(200, max_timestep=3000)
        dynaQplus_rewards.append(np.cumsum(dynaQplus_NF.rewards[:3000]))
        
    for i in range(30):
        blockedMaze.reset()
        dynaQplus = DynaQPlus_no_Footnote(blockedMaze, 0.8, 0.1, 100, 1e-4, 0.95)
        dynaQplus.train(200, max_timestep=3000)
        dynaQplus_NoF_rewards.append(np.cumsum(dynaQplus.rewards[:3000]))
    
    plt.plot(range(3000), np.mean(dynaQ_rewards, axis=0), label='dynaQ')
    plt.plot(range(3000), np.mean(dynaQplus_rewards, axis=0), label='dynaQ+')
    plt.plot(range(3000), np.mean(dynaQplus_NoF_rewards, axis=0), label='dynaQ+_no_footnote')
    plt.legend()
    plt.title('Figure 8.4_no_footnote')
    plt.xlabel('time steps')
    plt.ylabel('cummulative reward')
    plt.show()
    



def Shortcut__Maze():
    
            
    shortcut_maze = ShortcutMaze()

    dynaQ_rewards = []
    dynaQplus_rewards = []
    for i in range(5):
        shortcut_maze.reset()
        dynaQ = DynaQ(shortcut_maze, 0.7, 0.1, 50, 0.95)
        dynaQ.train(800, max_timestep=6000)
        dynaQ_rewards.append(np.cumsum(dynaQ.rewards[:6000]))
    
    for i in range(5):
        shortcut_maze.reset()
        dynaQplus = DynaQPlus(shortcut_maze, 0.7, 0.1, 50, 1e-3, 0.95)
        dynaQplus.train(800, max_timestep=6000)
        dynaQplus_rewards.append(np.cumsum(dynaQplus.rewards[:6000]))
    
    plt.plot(range(6000), np.mean(dynaQ_rewards, axis=0), label='dynaQ')
    plt.plot(range(6000), np.mean(dynaQplus_rewards, axis=0), label='dynaQ+')
    plt.legend()
    plt.title('Figure 8.5')
    plt.xlabel('time steps')
    plt.ylabel('cummulative reward')
    
    
    
def Shortcut__Maze_no_footnote():
    
            
    shortcut_maze = ShortcutMaze()

    dynaQ_rewards = []
    dynaQplus_rewards = []
    dynaQplus_NoF_rewards = []
    
    
    for i in range(5):
        shortcut_maze.reset()
        dynaQ = DynaQ(shortcut_maze, 0.7, 0.1, 50, 0.95)
        dynaQ.train(800, max_timestep=6000)
        dynaQ_rewards.append(np.cumsum(dynaQ.rewards[:6000]))
    
    for i in range(5):
        shortcut_maze.reset()
        dynaQplus = DynaQPlus(shortcut_maze, 0.7, 0.1, 50, 1e-3, 0.95)
        dynaQplus.train(800, max_timestep=6000)
        dynaQplus_rewards.append(np.cumsum(dynaQplus.rewards[:6000]))
        
        
    for i in range(5):
        shortcut_maze.reset()
        dynaQplus_NF = DynaQPlus_no_Footnote(shortcut_maze, 0.7, 0.1, 50, 1e-3, 0.95)
        dynaQplus_NF.train(800, max_timestep=6000)
        dynaQplus_NoF_rewards.append(np.cumsum(dynaQplus_NF.rewards[:6000]))    
    
    plt.plot(range(6000), np.mean(dynaQ_rewards, axis=0), label='dynaQ')
    plt.plot(range(6000), np.mean(dynaQplus_rewards, axis=0), label='dynaQ+')
    plt.plot(range(6000), np.mean(dynaQplus_NoF_rewards, axis=0), label='dynaQ+_no_footnote')
    plt.legend()
    plt.title('Figure 8.5_no_footnote')
    plt.xlabel('time steps')
    plt.ylabel('cummulative reward')


if __name__ == "__main__":
    Blocked_Maze()
    #Shortcut__Maze()
    #Blocked_Maze_no_footnote()
    #Shortcut__Maze_no_footnote()
    