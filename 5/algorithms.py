import gym
from typing import Callable, Tuple
from typing import Optional
from collections import defaultdict
import numpy as np
from env import register_env
import matplotlib.pyplot as plt

def generate_episodes(env, policy, num_episodes):
    episodes = []
    for i in range(num_episodes):
        episode = []
        state = env.reset()
        action = policy(state)
        episode.append((state, action, 0)) # R0 doesn't exist?
        while True:
            next_state, reward, done, _ = env.step(action)
            state = next_state
            action = policy(state)
            episode.append((state, action, reward))
            if done:
                episodes.append(episode)
                break
                
    return episodes

def create_epsilon_policy(Q: defaultdict, epsilon: float) -> Callable:
    """ an epsilon soft policy from Q values.

    """
    # Get number of actions
    num_actions = len(Q[0])

    def get_action(state: Tuple) -> int:
        # Make sure to break ties arbitrarily
        if np.random.random() < epsilon:
            action = np.random.randint(num_actions)
        else:
            action = np.random.choice(np.argwhere(Q[state] == np.amax(Q[state])).flatten())

        return action

    return get_action


def on_policy_mc_control_epsilon_soft(
    env: gym.Env, num_steps: int, gamma: float, epsilon: float
):
    """On-policy Monte Carlo policy control for epsilon soft policies.


    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    algorithm_returns = defaultdict(list)
    
    
    episodes_done = 0
    episodes_list = np.zeros(num_steps+1)
    s = 0 # number of steps so far
    get_action = create_epsilon_policy(Q, epsilon)
    state = env.reset()
    action = get_action(state)
    episode = []
    
    while True:
        next_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward))        
        state = next_state
        action = get_action(state)
        
        s += 1
        episodes_done += done # pretty cool that this works
        episodes_list[s] = episodes_done
                
        if done: # train and reset environment
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
                    
            state = env.reset()
            action = get_action(state)
            episode = []
            
        if s == num_steps:
            break # done completely
            
    return Q, episodes_list



def sarsa(env: gym.Env, num_steps: int, gamma: float, epsilon: float, step_size: float):
 

    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    episodes_done = 0
    episodes_list = np.zeros(num_steps+1)
    s = 0 # number of steps so far
    get_action = create_epsilon_policy(Q, epsilon)
    state = env.reset()
    action = get_action(state)
    
    while True:
        next_state, reward, done, _ = env.step(action)
        next_action = get_action(next_state)
        Q[state][action] = Q[state][action] + step_size * (reward + gamma * Q[next_state][next_action] - Q[state][action])
        
        state = next_state
        action = next_action
        s += 1
        episodes_done += done # pretty cool that this works
        episodes_list[s] = episodes_done
        
        if done: # reset env, time to train
            state = env.reset()
            action = get_action(state)
            
        if s == num_steps:
            break # done completely
                
    return Q, episodes_list


def nstep_sarsa( env: gym.Env, num_steps: int, gamma: float, epsilon: float, step_size: float):
  
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    episodes_done = 0
    episodes_list = np.zeros(num_steps+1)
    step_memory = []
    s = 0 # number of steps so far
    n = 2 # number of backup steps
    get_action = create_epsilon_policy(Q, epsilon)
    num_actions = len(Q[0])
    state = env.reset()
    action = get_action(state)
    step_memory.append((state,action, 0))
    t = 0 # number of steps in episode
    T = num_steps + 100000
    
    while True:
        if t < T:
            next_state, reward, done, _ = env.step(action)
            next_action = get_action(next_state)
            step_memory.append((next_state,next_action,reward))
            s += 1
            if done:
                T = t+1
                            
            state = next_state
            action = next_action
            
            episodes_done += done # pretty cool that this works
            episodes_list[s] = episodes_done
        
        tau = t - n + 1
        if tau >= 0:
            
            loop_limit = min(tau + n, T)
            rho = 1
            for i in range(tau+1,loop_limit+1):
                # need policy probabilities. this sucks
                i_state = step_memory[i][0]
                i_action = step_memory[i][1]
                # greedy will be 0 or 1 (unless there are ties...)
                optimal_actions = np.argwhere(Q[i_state] == np.amax(Q[i_state])).flatten()
                if i_action in optimal_actions: # greedy was taken
                    greedy_prob = 1 / len(optimal_actions)
                    policy_probability = ((epsilon / num_actions) + (1-epsilon) / len(optimal_actions))
                else:
                    greedy_prob = 0
                    policy_probability = (epsilon / num_actions)
                    
                rho *= greedy_prob / policy_probability
                
            loop_limit = min(tau + n, T)
            G = 0
            for i in range(tau+1,loop_limit):
                reward = step_memory[i][2]
                G += gamma**(i - tau - 1) * reward
                
            if tau + n < T:
                tau_n_state = step_memory[tau+n][0]
                tau_n_action = step_memory[tau+n][1]
                G = G + gamma**n * Q[tau_n_state][tau_n_action]
                
                
            tau_state = step_memory[tau][0]
            tau_action = step_memory[tau][1]
            Q[tau_state][tau_action] = Q[tau_state][tau_action] + step_size * rho * (G - Q[tau_state][tau_action])
        
        t += 1
        
        if done: # reset env
            state = env.reset()
            action = get_action(state)
            step_memory = []
            step_memory.append((state,action, 0))
            t = 0 # number of steps in episode
            T = num_steps + 100000
            
        if s == num_steps:
            break # done completely
                
    return Q, episodes_list


def exp_sarsa(env: gym.Env,num_steps: int,gamma: float,epsilon: float,step_size: float):
    """Expected SARSA
    """

    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    episodes_done = 0
    episodes_list = np.zeros(num_steps+1)
    s = 0 # number of steps so far
    get_action = create_epsilon_policy(Q, epsilon)
    num_actions = len(Q[0])
    state = env.reset()
    action = get_action(state)
    
    while True:
        next_state, reward, done, _ = env.step(action)
        
        # get target through expected values
        optimal_actions = np.argwhere(Q[next_state] == np.amax(Q[next_state]))
        expected_target = 0
        # probability of a suboptimal action = e / num actions
        # probability of an optimal action = e / num actions + (1-e) / num optimal actions <- in case of ties
        for next_a in range(num_actions):
            if next_a not in optimal_actions:
                expected_target += (epsilon / num_actions) * Q[next_state][next_a]
            else:
                expected_target+= ((epsilon / num_actions) + (1-epsilon) / len(optimal_actions)) * Q[next_state][next_a]
        
        
        Q[state][action] = Q[state][action] + step_size * (reward + gamma * expected_target - Q[state][action])
        
        # next action picked same way as Q-learning
        state = next_state
        action = get_action(state)
        s += 1
        episodes_done += done # pretty cool that this works
        episodes_list[s] = episodes_done
        
        if done: # reset env
            state = env.reset()
            action = get_action(state)
            
        if s == num_steps:
            break # done completely
                
    return Q, episodes_list


def q_learning(env: gym.Env,num_steps: int,gamma: float,epsilon: float,step_size: float):
  
    
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    episodes_done = 0
    episodes_list = np.zeros(num_steps+1)
    s = 0 # number of steps so far
    get_action = create_epsilon_policy(Q, epsilon)
    state = env.reset()
    action = get_action(state)
    
    while True:
        next_state, reward, done, _ = env.step(action)
        target_action = np.random.choice(np.argwhere(Q[next_state] == np.amax(Q[next_state])).flatten())
        Q[state][action] = Q[state][action] + step_size * (reward + gamma * Q[next_state][target_action] - Q[state][action])
        
        state = next_state
        action = get_action(state)
        s += 1
        episodes_done += done # pretty cool that this works
        episodes_list[s] = episodes_done
        
        if done: # reset env
            state = env.reset()
            action = get_action(state)
            
        if s == num_steps:
            break # done completely
                
    return Q, episodes_list


def td_prediction(gamma: float, episodes, step_size: float, n=1) -> defaultdict:

    V = defaultdict(float)
    for episode in episodes:
        T = len(episode) + n + 50 # make T big enough I guess
        for t in range(len(episode)):
            # episode generation already done
            if t + 1 == len(episode) - 1: # terminal step
                T = t + 1
            tau = t - n + 1
            if tau >= 0:
                loop_max = min(tau+n,T)
                G = 0
                for i in range(tau + 1, loop_max+1): # doesn't loop if n == 1 though...
                    reward_i = episode[i][2]
                    G += gamma**(i - tau - 1) * reward_i
                
                if tau + n < T:
                    state_tau_n = episode[tau+n][0]
                    G += gamma**n * V[state_tau_n]
                
                state_tau = episode[tau][0]    
                V[state_tau] = V[state_tau] + step_size * (G - V[state_tau])
    return V
    
def mc_evaluation(gamma: float, episodes) -> defaultdict:
    """Monte Carlo policy evaluation. First visits will be used.

    """
    # We use defaultdicts here for both V and N for convenience. The states will be the keys.
    V = defaultdict(float)
    N = defaultdict(int)
    returns = defaultdict(list)

    for episode in episodes:
        # store earliest appearances of states
        appearance = defaultdict(int)
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            appearance[state] = t
        
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward
            
            if appearance[state] == t: # we are at first appearance
                N[state] += 1
                returns[state].append(G)
                V[state] = sum(returns[state])/N[state]
            
    return V

def learning_targets(
    V: defaultdict, gamma: float, episodes, n: Optional[int] = None
) -> np.ndarray:
    """Compute the learning targets for the given evaluation episodes.

    """
    targets = np.zeros(len(episodes))
    if n == None: # monte carlo
        for i, episode in enumerate(episodes):
            G = 0
            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G = gamma * G + reward
            targets[i] = G
    else: # TD
        for i, episode in enumerate(episodes):
            loop_max = min(n+1,len(episode))
            G = 0
            
            for t in range(loop_max):
                state, action, reward = episode[t]
                # we only need to take n steps for each episode
                G += gamma**t * reward
                if t == loop_max -1: # final step, add V(S')
                    G += gamma**loop_max * V[episode[t+1][0]]
                
            targets[i] = G
        
    return targets
    
def four_b_main():
    register_env()
    # check for passing non (10,10) env's    
    env = gym.make("WindyGridWorld-v0")
    
    num_episodes = 10000
    num_trials = 10
    num_steps = 8000
    gamma = 1
    epsilon = 0.1
    step_size = 0.5
    
    mc_eps = np.asarray([on_policy_mc_control_epsilon_soft(env, num_steps, gamma, epsilon)[1] for i in range(num_trials)])
    sarsa_eps = np.asarray([sarsa(env, num_steps, gamma, epsilon, step_size)[1] for i in range(num_trials)])
    q_eps = np.asarray([q_learning(env, num_steps, gamma, epsilon, step_size)[1] for i in range(num_trials)])
    exp_sarsa_eps = np.asarray([exp_sarsa(env, num_steps, gamma, epsilon, step_size)[1] for i in range(num_trials)])
    nstep_sarsa_eps = np.asarray([nstep_sarsa(env, num_steps, gamma, epsilon, step_size)[1] for i in range(num_trials)])
    eps = [mc_eps, sarsa_eps, q_eps, exp_sarsa_eps, nstep_sarsa_eps]
    labels = ['Monte-Carlo','SARSA','Q-Learning','Expected SARSA','N-Step SARSA (n=2)']
    for i,ep in enumerate(eps):
        mean = np.mean(ep,axis=0)
        std_dev = np.std(ep, axis = 0)
        std_err = std_dev / (num_trials ** (1/2))
        interval = 1.96 * std_err
        x = np.arange(0, 8001)
        plt.fill_between(x, y1 = mean - interval, y2 = mean + interval, alpha=0.5)
        plt.plot(mean, label=labels[i])
    
    plt.legend()
    plt.ylabel("Episodes")
    plt.xlabel("Time steps")    
    plt.show()

    
def four_c_main():
    register_env()
    # check for passing non (10,10) env's    
    env = gym.make("WindyGridWorld-v0")
    king_env = gym.make("WindyGridWorldKings-v0")
    king_stay_env = gym.make("WindyGridWorldKings-v1")
    
    num_episodes = 10000
    num_trials = 10
    num_steps = 8000
    gamma = 1
    epsilon = 0.1
    step_size = 0.5
    
    q, sarsa_eps = sarsa(env, num_steps, gamma, epsilon, step_size)
    q, q_eps = q_learning(env, num_steps, gamma, epsilon, step_size)
    q, king_sarsa_eps = sarsa(king_env, num_steps, gamma, epsilon, step_size)
    q, king_q_eps = q_learning(king_env, num_steps, gamma, epsilon, step_size)
    q, king_sarsa_stay_eps = sarsa(king_stay_env, num_steps, gamma, epsilon, step_size)
    q, king_q_stay_eps = q_learning(king_stay_env, num_steps, gamma, epsilon, step_size)
        
    
    
    plt.plot(sarsa_eps, 'b', label='Regular SARSA', markersize=1)
    plt.plot(q_eps, 'r', label='Regular Q')
    plt.plot(king_sarsa_eps, 'b--', label='King SARSA', markersize=1)
    plt.plot(king_q_eps, 'r--', label='King Q')
    plt.plot(king_sarsa_stay_eps, 'b:', label='King SARSA (with Stay)', markersize=1)
    plt.plot(king_q_stay_eps, 'r:', label='King Q (with Stay)')
    plt.legend()
    plt.title("Comparison with King's Moves")
    plt.ylabel("Episodes")
    plt.xlabel("Time steps")
    plt.show()
    
def four_d_main():
    register_env()
    # check for passing non (10,10) env's    
    env = gym.make("WindyGridWorld-v0")
    king_env = gym.make("WindyGridWorldStochastic-v0")
    king_stay_env = gym.make("WindyGridWorldStochastic-v1")
    
    num_episodes = 10000
    num_trials = 10
    num_steps = 8000
    gamma = 1
    epsilon = 0.1
    step_size = 0.5
    
    q, sarsa_eps = sarsa(env, num_steps, gamma, epsilon, step_size)
    q, q_eps = q_learning(env, num_steps, gamma, epsilon, step_size)
    q, king_sarsa_eps = sarsa(king_env, num_steps, gamma, epsilon, step_size)
    q, king_q_eps = q_learning(king_env, num_steps, gamma, epsilon, step_size)
    q, king_sarsa_stay_eps = sarsa(king_stay_env, num_steps, gamma, epsilon, step_size)
    q, king_q_stay_eps = q_learning(king_stay_env, num_steps, gamma, epsilon, step_size)
        
    
    
    plt.plot(sarsa_eps, 'b', label='Regular SARSA', markersize=1)
    plt.plot(q_eps, 'r', label='Regular Q')
    plt.plot(king_sarsa_eps, 'b--', label='King SARSA', markersize=1)
    plt.plot(king_q_eps, 'r--', label='King Q')
    plt.plot(king_sarsa_stay_eps, 'b:', label='King SARSA (with Stay)', markersize=1)
    plt.plot(king_q_stay_eps, 'r:', label='King Q (with Stay)')
    plt.legend()
    plt.title("Comparison with Stochastic Wind")
    plt.ylabel("Episodes")
    plt.xlabel("Time steps")
    plt.show()
        
def five_a_main():
    register_env()
    env = gym.make("WindyGridWorld-v0")
    
    num_steps = 8000
    gamma = 1
    epsilon = 0.1
    step_size = 0.5
    
    # q gives us the q values needed for the policy
    q, mc_eps = sarsa(env, num_steps, gamma, epsilon, step_size)
    get_action = create_epsilon_policy(q, epsilon)
    # next we generate training and eval episodes
    episodes_1 = generate_episodes(env, get_action, 1)
    episodes_10 = generate_episodes(env, get_action, 10)
    episodes_50 = generate_episodes(env, get_action, 50)
                    
    eval_episodes = generate_episodes(env, get_action, 100)
    
    value_funcs = [] # store the 9 value functions in order: monte, TD(0), TD(N)
    value_funcs.append(mc_evaluation(gamma, episodes_1))
    value_funcs.append(mc_evaluation(gamma, episodes_10))
    value_funcs.append(mc_evaluation(gamma, episodes_50))
    value_funcs.append(td_prediction(gamma, episodes_1, step_size, n=1))
    value_funcs.append(td_prediction(gamma, episodes_10, step_size, n=1))
    value_funcs.append(td_prediction(gamma, episodes_50, step_size, n=1))
    value_funcs.append(td_prediction(gamma, episodes_1, step_size, n=4))
    value_funcs.append(td_prediction(gamma, episodes_10, step_size, n=4))
    value_funcs.append(td_prediction(gamma, episodes_50, step_size, n=4))
    
    ns = [None, None, None, 1, 1, 1, 4, 4, 4]
    targets = []
    for i, v_func in enumerate(value_funcs):
        targets.append(learning_targets(v_func, gamma, eval_episodes, n=ns[i]))
    
    labels = ["MC: N=1","MC: N=10","MC: N=50","TD(0): N=1","TD(0): N=10","TD(0): N=50","TD(4): N=1","TD(4): N=10","TD(4): N=50"]
    fig, axs = plt.subplots(3, 3)
    for y in range(3):
        for x in range(3):
            flattened_index = x + y*3
            axs[y,x].hist(targets[flattened_index])
            axs[y,x].set_title(labels[flattened_index])
            
    fig.tight_layout()
    plt.show()
        

if __name__ == "__main__":
    four_b_main()
    #four_c_main()
    #four_d_main()
    #five_a_main()
