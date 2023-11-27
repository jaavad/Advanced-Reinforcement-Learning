import gym
from typing import Callable, Tuple
from collections import defaultdict
from tqdm import trange
import numpy as np
from policy import create_blackjack_policy, create_epsilon_policy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from env import register_env, FourRoomsEnv


def generate_episode(env: gym.Env, policy: Callable, es: bool = False):
    """A function to generate one episode and collect the sequence of (s, a, r) tuples

    This function will be useful for implementing the MC methods

    Args:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
        es (bool): Whether to use exploring starts or not
    """
    episode = []
    state = env.reset()
    while True:
        if es and len(episode) == 0:
            action = env.action_space.sample()
        else:
            action = policy(state)

        next_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward))
        if done:
            break
        state = next_state

    return episode


def on_policy_mc_evaluation(
    env: gym.Env,
    policy: Callable,
    num_episodes: int,
    gamma: float,
) -> defaultdict:
    """On-policy Monte Carlo policy evaluation. First visits will be used.

    Args:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP

    Returns:
        V (defaultdict): The values for each state. V[state] = value.
    """
    # We use defaultdicts here for both V and N for convenience. The states will be the keys.
    V = defaultdict(float)
    N = defaultdict(int)
    returns = defaultdict(list)

    for _ in trange(num_episodes, desc="Episode"):
        episode = generate_episode(env, policy)
        # store earliest appearances of states
        appearance = defaultdict(int)
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            appearance[state] = t
        
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            # TODO Q3a
            state, action, reward = episode[t]
            G = gamma * G + reward
            
            if appearance[state] == t: # we are at first appearance
                N[state] += 1
                returns[state].append(G)
                V[state] = sum(returns[state])/N[state]
            
    return V


def on_policy_mc_control_es(
    env: gym.Env, num_episodes: int, gamma: float
) -> Tuple[defaultdict, Callable]:
    """On-policy Monte Carlo control with exploring starts for Blackjack

    Args:
        env (gym.Env): a Gym API compatible environment
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP
    """
    # We use defaultdicts here for both Q and N for convenience. The states will be the keys and the values will be numpy arrays with length = num actions
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    returns = defaultdict(list)

    # If the state was seen, use the greedy action using Q values.
    # Else, default to the original policy of sticking to 20 or 21.
    policy = create_blackjack_policy(Q)

    for _ in trange(num_episodes, desc="Episode"):
        # TODO Q3b
        # Note there is no need to update the policy here directly.
        # By updating Q, the policy will automatically be updated.
        episode = generate_episode(env, policy, es=True)
        # store earliest appearances of states
        appearance = defaultdict(int)
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            appearance[state + (action,)] = t
        
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward
            
            if appearance[state + (action,)] == t: # we are at first appearance
                N[state][action] += 1
                returns[state + (action,)].append(G)
                Q[state][action] = np.sum(returns[state + (action,)])/N[state][action]
                # policy update step not necessary

    return Q, policy


def on_policy_mc_control_epsilon_soft(
    env: gym.Env, num_episodes: int, gamma: float, epsilon: float
):
    """On-policy Monte Carlo policy control for epsilon soft policies.

    Args:
        env (gym.Env): a Gym API compatible environment
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP
        epsilon (float): Parameter for epsilon soft policy (0 <= epsilon <= 1)
    Returns:

    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    algorithm_returns = defaultdict(list)
    
    policy = create_epsilon_policy(Q, epsilon)

    returns = np.zeros(num_episodes)
    for i in trange(num_episodes, desc="Episode", leave=False):
        # TODO Q4
        # For each episode calculate the return
        # Update Q
        # Note there is no need to update the policy here directly.
        # By updating Q, the policy will automatically be updated.
        episode = generate_episode(env, policy)
        # store earliest appearances of states
        appearance = defaultdict(int)
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            appearance[state + (action,)] = t
        
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward
            
            if appearance[state + (action,)] == t: # we are at first appearance
                N[state][action] += 1
                algorithm_returns[state + (action,)].append(G)
                Q[state][action] = np.sum(algorithm_returns[state + (action,)])/N[state][action]
                # policy update step not necessary
                
        returns[i] = G

    return returns

# use this function to generate  surface map for plotting purposes
def generate_surface_map(values, aces):
    X = np.arange(1,11) # dealer showing
    Y = np.arange(12,22) # player sum
    surface_map = np.zeros((Y.size,X.size))
    
    for x in X:
        for y in Y:
            surface_map[y-12,x-1] = values[(y,x,aces)]
    
    return surface_map

def three_a_main():
    env = gym.make('Blackjack-v1')
    # pass empty dict so policy just uses default
    policy = create_blackjack_policy({})
    num_episodes = 10000
    gamma = 1
    # generate Value functions
    V1 = on_policy_mc_evaluation(env, policy, num_episodes, gamma)
    num_episodes = 500000
    V2 = on_policy_mc_evaluation(env, policy, num_episodes, gamma)
    # separate into usable aces and not

    x = np.arange(1,11) # dealer showing
    y = np.arange(12,22) # player sum
    X, Y = np.meshgrid(x,y)

    Z1_no_aces = generate_surface_map(V1,False)
    Z1_aces = generate_surface_map(V1,True)
    Z2_no_aces = generate_surface_map(V2,False)
    Z2_aces = generate_surface_map(V2,True)

    fig1 = plt.figure()
    ax = fig1.add_subplot(1,2,1, projection='3d')
    ax.plot_surface(X, Y, Z1_aces, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
    ax = fig1.add_subplot(1,2,2, projection='3d')
    ax.plot_surface(X, Y, Z2_aces, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
    ax.set_zlabel('V')
    
    
    fig1.supylabel("Usable Ace")
    plt.show()
    
    fig2 = plt.figure()
    ax = fig2.add_subplot(1,2,1, projection='3d')
    ax.plot_surface(X, Y, Z2_no_aces, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
    ax = fig2.add_subplot(1,2,2, projection='3d')
    ax.plot_surface(X, Y, Z2_no_aces, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
    ax.set_xlabel('Dealer Showing')
    ax.set_ylabel('Player Sum')
    
    fig2.supylabel("No Usable Ace")
    fig2.supxlabel("After 10,000 and After 500,000 Episodes")
    plt.show()
    
def three_b_main():
    env = gym.make('Blackjack-v1')
    # pass empty dict so policy just uses default
    policy = create_blackjack_policy({})
    num_episodes = 3000000
    gamma = 1
    # generate Q and policy
    Q, policy = on_policy_mc_control_es(env, num_episodes, gamma)
    
    # generate value function
    V = {key : np.max(Q[key]) for key in Q.keys()}
    Policy_Map = {key : np.argmax(Q[key]) for key in Q.keys()}
    
    
    # separate into usable aces and not

    x = np.arange(1,11) # dealer showing
    y = np.arange(12,22) # player sum
    X, Y = np.meshgrid(x,y)

    Z_no_aces = generate_surface_map(V,False)
    Z_aces = generate_surface_map(V,True)
    policy_no_aces = generate_surface_map(Policy_Map,False)
    policy_aces = generate_surface_map(Policy_Map,True)
    
    fig1 = plt.figure()
    ax = fig1.add_subplot(1,2,1)
    ax.contourf(X, Y, policy_aces,levels=1)
    ax.set_xlabel('Dealer Showing')
    ax.set_ylabel('Player Sum')
    ax = fig1.add_subplot(1,2,2, projection='3d')
    ax.plot_surface(X, Y, Z_aces, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
    ax.set_zlabel('V')
    
    
    fig1.supylabel("Usable Ace")
    plt.show()
    
    fig2 = plt.figure()
    ax = fig2.add_subplot(1,2,1)
    ax.contourf(X, Y, policy_no_aces,levels=1)
    ax = fig2.add_subplot(1,2,2, projection='3d')
    ax.plot_surface(X, Y, Z_no_aces, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
    ax.set_xlabel('Dealer Showing')
    ax.set_ylabel('Player Sum')
    
    fig2.supylabel("No Usable Ace")
    fig2.supxlabel("Optimal Policy and Value Function")
    plt.show()
    
def four_main():
    register_env()
    # check for passing non (10,10) env's
    test_env = gym.make('FourRooms-v0', goal_pos=(3,7))
    
    env = gym.make('FourRooms-v0')

    num_episodes = 10000
    num_trials = 10
    gamma = 0.99
    epsilons = [0.01, 0.1, 0]
    labels = ["epsilon=0.1","epsilon=0.01","epsilon=0"]
    
    test_returns = on_policy_mc_control_epsilon_soft(test_env, num_episodes, gamma, 0.1)
    
    plt.plot(test_returns)
    plt.title("Returns of (4,8) goal position")
    plt.ylabel("Return")
    plt.xlabel("Episode")
    plt.show()
    
    # generate returns
    returns1 = np.zeros((num_trials,num_episodes))
    returns2 = np.zeros((num_trials,num_episodes))
    returns3 = np.zeros((num_trials,num_episodes))
    for t in range(3):
        returns = np.zeros((num_trials,num_episodes))
        for i in range(num_trials):
            returns[i] = on_policy_mc_control_epsilon_soft(env, num_episodes, gamma, epsilons[t])
        
        mean = np.mean(returns,axis=0)
        std_dev = np.std(returns, axis = 0)
        std_err = std_dev / (num_trials ** (1/2))
        interval = 1.96 * std_err
        x = np.arange(0, num_episodes)
        
        plt.fill_between(x, y1 = mean - interval, y2 = mean + interval, alpha=0.5)
        plt.plot(mean,label=labels[t])
    
    # theoretical limit
    limit = gamma**(10+10)
    plt.hlines(limit,0,num_episodes,colors="r")
    
    plt.legend()
    plt.ylabel("Mean Return")
    plt.xlabel("Episode")
    plt.show()
    

if __name__ == "__main__":
    #three_a_main()
    #three_b_main()
    four_main()