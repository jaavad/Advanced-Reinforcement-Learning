# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 00:46:25 2022

@author: Siavash
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
from env import register_env, FourRoomsEnv
from collections import defaultdict



def generate_episode (env: gym.Env, Policy: Callable, max_steps: int):
    
    
    current_state=env.reset()
    
    i=0
    episod=
    cummulative_return =0
    
    while i < max_steps:
                  
        action = Policy(current_state)       
        next_state,reward,done,_ = env(current_state, action)        
        episod[i] = [current_stat,action,reward]
        current_stat = next_state
        Ap[current_stat][action]==i
        cummulative_return += reward
        i+=1 
        
        if done:
            break
        
    return episod,cummulative_return
        
      
def e_soft_policy (Q:difauldict, e:):
        
    if np.random < e or ???:
        policy[:] = Q[:][random]
        
        else:
            policy[:] = Q[:][argmax()]
    
    def take_action (state:Tuple(int,int),policy):     
        action = policy[state]
        
        return action
    
    return policy 



def SARSA (evn: gym.Env, e_soft_policy:callable,num_episodes: int,max_steps:int,gamma:float, alpha: float):
    
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    V = defaultdict(float)
    N = defaultdict(int)
    returns = defaultdict(list)
    
    
    for j in range(num_episod):   
            
        same seed
        policy = e_soft_policy (Q,e)
      
        while i < max_steps:
                      
            action = Policy(current_state)       
            next_state,reward,done,_ = env(current_state, action)        
            next_action = Policy(next_state)       

            Q[current_state][action] = Q[current_state][action] + alpha*()

            Ap[current_stat][action]==i
            cummulative_return += reward
            i+=1 
            
            if done:
                break
            
        return episod,cummulative_return
      
        
      episod, cummulative_return,Ap= generate_episode(env,policy,max_steps)
        size_episod = np.shape(episod[0])
        cummulative_returns[i,j]=cummulative_return
        
        
        def update_Q (episod,)
        
        G=0
        
        for k in range(size_episod),-1:
            
                state,action,reward = episode(k)
            
                G = gamma*G + reward
           
                if FA:
                
                    if Ap[state][action]==k:
                    
                        returnn[state][action].append(G)            
                        N [state][action]+=1
                        Q[state][action] = sum(returnn[state][action])/N [state][action]
        





def Q4 ():
    
    
    register_env()
    # check for passing non (10,10) env's
    test_env = gym.make('FourRooms-v0', goal_pos=(3,7))
    
    env = gym.make('FourRooms-v0')


    num_episodes=
    max_steps=
    num_trials=
    gamma=

    for i in range(num_trials):

       cummulative_returns, policy, value, q_value = on_policy_mc_control_epsilon_soft_agent (evn,e_soft_policy,num_episodes,max_steps,gamma)
       
       ## plot cummulative_returns
        ## plot policy surface 
        ## plot value
        ## plot q value  
if __name__ == "__main__":
    
    Q4()        
        
        
        