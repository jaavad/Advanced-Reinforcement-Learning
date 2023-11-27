import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import copy


class DynaQ:
    def __init__(self, env, alpha, epsilon, n, gamma=1):
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = defaultdict(int)
        self.model = {}
        self.actions = env.actions
        self.n = n
        self.stats = [] # each entry will be a tuple of (episode, steps per episode)
        self.archive = []
        self.rewards = []
        
    def select_action(self, state):
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            best_value = -10000
            best_action = []
            for action in self.actions:
                value = self.Q[state, action]
                if value > best_value:
                    best_value = value
                    best_action = [action]
                elif value == best_value:
                    best_action.append(action)
            return np.random.choice(best_action)
        
    def best_Q(self, state):
        return max([self.Q[state, action] for action in self.actions])
    
    def train(self, episodes, max_timestep = None):
        for episode in range(1, 1 + episodes):
            steps = 0
            state = self.env.start()
            is_end = False
            while not is_end:
                action = self.select_action(state)
                next_state, reward, is_end = self.env.act(action)
                self.Q[state, action] += self.alpha * (reward + self.gamma * self.best_Q(next_state) - self.Q[state, action])
                self.model[state, action] = (reward, next_state)
                state = next_state
                self.rewards.append(reward)
                
                #### Planning phase
                for i in range(self.n):
                    sample_state, sample_action = list(self.model.keys())[np.random.randint(0, len(self.model))]
                    sample_reward, sample_next_state = self.model[sample_state, sample_action]
                    self.Q[sample_state, sample_action] += self.alpha * (sample_reward + \
                                        self.gamma * self.best_Q(sample_next_state) - self.Q[sample_state, sample_action])
                steps += 1
                if max_timestep is not None and len(self.rewards) > max_timestep:
                    return
                if episode == 2:
                    # archive
                    self.archive.append((state, copy.deepcopy(self.Q)))
            self.stats.append((episode, steps))
            
            
            
            
            
class DynaQPlus:
    def __init__(self, env, alpha, epsilon, n, kappa, gamma=1):
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = defaultdict(int)
        self.model = {}
        self.actions = env.actions
        self.n = n
        self.kappa = kappa
        self.rewards = []
        self.steps = 0
        self.visited = {}
        self.visited_states = []
        
    def select_action(self, state):
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            best_value = -10000
            best_action = []
            for action in self.actions:
                value = self.Q[state, action]
                if value > best_value:
                    best_value = value
                    best_action = [action]
                elif value == best_value:
                    best_action.append(action)
            return np.random.choice(best_action)
        
    def best_Q(self, state):
        return max([self.Q[state, action] for action in self.actions])
    
    def train(self, episodes, max_timestep = None):
        for episode in range(1, 1 + episodes):
            state = self.env.start()
            is_end = False
            while not is_end:
                action = self.select_action(state)
                next_state, reward, is_end = self.env.act(action)
                self.Q[state, action] += self.alpha * (reward + self.gamma * self.best_Q(next_state) - self.Q[state, action])
                self.model[state, action] = (reward, next_state, self.steps)
               
                for a in self.actions:
                    if not a == action and (state, a) not in self.model.keys():
                        self.model[(state, a)] = (0, state, 0)
                        
                state = next_state
                self.rewards.append(reward)
                
                ##### Planning phase
                for i in range(self.n):
                    sample_state, sample_action = list(self.model.keys())[np.random.randint(0, len(self.model))]
                    sample_reward, sample_next_state, time = self.model[sample_state, sample_action]
                    sample_reward += self.kappa * np.sqrt(self.steps - time)
                    self.Q[sample_state, sample_action] += self.alpha * (sample_reward + \
                                        self.gamma * self.best_Q(sample_next_state) - self.Q[sample_state, sample_action])
                self.steps += 1
                if max_timestep is not None and self.steps > max_timestep:
                    return 
        
        
        
class DynaQPlus_no_Footnote:
    def __init__(self, env, alpha, epsilon, n, kappa, gamma=1):
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = defaultdict(int)
        self.model = {}
        self.actions = env.actions
        self.n = n
        self.kappa = kappa
        self.rewards = []
        self.steps = 0
        self.visited = {}
        self.visited_states = []
        
    def select_action(self, state):
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            best_value = -10000
            best_action = []
            for action in self.actions:
                value = self.Q[state, action]
                if value > best_value:
                    best_value = value
                    best_action = [action]
                elif value == best_value:
                    best_action.append(action)
            return np.random.choice(best_action)
        
    def best_Q(self, state):
        return max([self.Q[state, action] for action in self.actions])
    
    def train(self, episodes, max_timestep = None):
        for episode in range(1, 1 + episodes):
            state = self.env.start()
            is_end = False
            while not is_end:
                action = self.select_action(state)
                next_state, reward, is_end = self.env.act(action)
                self.Q[state, action] += self.alpha * (reward + self.gamma * self.best_Q(next_state) - self.Q[state, action])
                self.model[state, action] = (reward, next_state, self.steps)
                state = next_state
                self.rewards.append(reward)
                
                ##### Planning phase
                for i in range(self.n):
                    sample_state, sample_action = list(self.model.keys())[np.random.randint(0, len(self.model))]
                    sample_reward, sample_next_state, time = self.model[sample_state, sample_action]
                    sample_reward += self.kappa * np.sqrt(self.steps - time)
                    self.Q[sample_state, sample_action] += self.alpha * (sample_reward + \
                                        self.gamma * self.best_Q(sample_next_state) - self.Q[sample_state, sample_action])
                self.steps += 1
                if max_timestep is not None and self.steps > max_timestep:
                    return 
        
        
                