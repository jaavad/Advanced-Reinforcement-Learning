import gym
import numpy as np
import matplotlib.pyplot as plt
from env import register_env, FourRoomsEnv

# state aggregate features
def single_aggregate_state_feature(state): # each state is only itself, feature vector = height*width
    x,y = state
    feature_vector = np.zeros(11*11)
    feature_vector[11*y+x] = 1
    return feature_vector

def row_aggregate(state): # aggregate all rows
    x,y = state
    feature_vector = np.zeros(11)
    feature_vector[y] = 1
    return feature_vector

def center_distance_aggregate(state): # aggregate by manhatten distance to center (0 to 10)
    x,y = state
    feature_vector = np.zeros(11)
    feature_vector[abs(x-5) + abs(y-5)] = 1
    return feature_vector

def goal_distance_aggregate(state): # aggregate by manhatten distance to goal (0 to 20)
    x,y = state
    feature_vector = np.zeros(21)
    feature_vector[abs(x-10) + abs(y-10)] = 1
    return feature_vector

# linear features
def linear_feature(state): # basic x/y/bias feature
    x,y = normalize_state(state)
    feature_vector = np.array([x,y,1])
    return feature_vector

def difference_feature(state): # difference between x and y, y and x
    x,y = normalize_state(state)
    feature_vector = np.array([x-y,y-x,1]) # no idea if this really makes sense
    return feature_vector

def squared_feature(state): # squared polynomial
    x,y = normalize_state(state)
    feature_vector = np.array([x**2,y**2,x*y,1])
    return feature_vector

def squared_feature_shifted(state): # squared polynomial
    x,y = normalize_state(state)
    x += .5
    y += .5
    feature_vector = np.array([x**2,y**2,x*y,1])
    return feature_vector

# helper functions
def get_epsilon_action(state, weights, num_actions, epsilon) -> int:
    if np.random.random() < epsilon:
        action = np.random.randint(num_actions)
    else:
        q_values = np.asarray([approximate_q_value(state, a, weights) for a in range(num_actions)])
        action = np.random.choice(np.argwhere(q_values == np.amax(q_values)).flatten())

    return action

def normalize_state(state):
    x,y = state
    x = (x-5)/10
    y = (y-5)/10
    return x,y

def approximate_q_value(feature, action, weights):
    q_value = (weights @ feature)[action]
    return q_value

def semi_grad_sarsa(env: gym.Env, num_eps: int, gamma: float, epsilon: float, step_size: float, state_approximator):
    episodes_list = np.zeros(num_eps)
    ep_length = 0

    def compute_gradient(state, action, weights):
        grad = np.zeros_like(weights)
        grad[action] = state_approximator(state)
        return grad

    state = env.reset()
    feature = state_approximator(state)
    num_actions = env.action_space.n
    weights = np.zeros((num_actions,len(feature)))
    action = get_epsilon_action(feature, weights, num_actions, epsilon)
    done = False

    for i in range(num_eps):
        while not done and ep_length < 2000: # loop for each episode, limit length cause it can get crazy with bad approximators
            ep_length += 1
            next_state, reward, done, _ = env.step(action)
            next_feature = state_approximator(next_state)

            next_action = get_epsilon_action(next_feature, weights, num_actions, epsilon)

            step_q = approximate_q_value(feature, action, weights)
            next_q = approximate_q_value(next_feature, next_action, weights)
            # Syed Asjad pointed out that my next_q and step_q were originally swapped. By fixing this, my algorithm worked
            weights += step_size * (reward + gamma * next_q - step_q) * compute_gradient(state, action, weights)

            feature = next_feature
            state = next_state
            action = next_action

        # this space reached once the episode is done
        weights += step_size * (reward - approximate_q_value(feature, action, weights)) * compute_gradient(state, action, weights)
        episodes_list[i] = ep_length
        ep_length = 0
        state = env.reset()
        feature = state_approximator(state)
        action = get_epsilon_action(feature, weights, num_actions, epsilon)
        done = False

    return weights, episodes_list

def main():
    register_env()
    env = gym.make('FourRooms-v0')

    # reduced num_trials for linear features due to time
    num_trials = 10
    num_eps = 100
    gamma = 0.95
    epsilon = 0.05
    step_size = 0.05

    # aggregate features
    #sarsa_eps = np.asarray([semi_grad_sarsa(env, num_eps, gamma, epsilon, step_size, single_aggregate_state_feature)[1] for i in range(num_trials)])
    #sarsa_eps = np.asarray([semi_grad_sarsa(env, num_eps, gamma, epsilon, step_size, row_aggregate)[1] for i in range(num_trials)])
    #sarsa_eps = np.asarray([semi_grad_sarsa(env, num_eps, gamma, epsilon, step_size, center_distance_aggregate)[1] for i in range(num_trials)])
    #sarsa_eps = np.asarray([semi_grad_sarsa(env, num_eps, gamma, epsilon, step_size, goal_distance_aggregate)[1] for i in range(num_trials)])
    
    # linear features
    step_size = 0.03 # different linear features work best with different learning rates
    #sarsa_eps = np.asarray([semi_grad_sarsa(env, num_eps, gamma, epsilon, 0.01, linear_feature)[1] for i in range(num_trials)])
    #sarsa_eps = np.asarray([semi_grad_sarsa(env, num_eps, gamma, epsilon, step_size, difference_feature)[1] for i in range(num_trials)])
    #sarsa_eps = np.asarray([semi_grad_sarsa(env, num_eps, gamma, epsilon, step_size, squared_feature)[1] for i in range(num_trials)])
    sarsa_eps = np.asarray([semi_grad_sarsa(env, num_eps, gamma, epsilon, 0.01, squared_feature_shifted)[1] for i in range(num_trials)])

    mean = np.mean(sarsa_eps,axis=0)
    std_dev = np.std(sarsa_eps, axis = 0)
    std_err = std_dev / (num_trials ** (1/2))
    interval = 1.96 * std_err
    x = np.arange(0, num_eps)
    plt.fill_between(x, y1 = mean - interval, y2 = mean + interval, alpha=0.5)
    plt.plot(mean)

    plt.ylabel("Time Steps")
    plt.xlabel("Episode")
    plt.title("Quadratic Shifted Approximator")
    plt.show()

if __name__ == "__main__":
    main()
