

from env import JacksCarRental
import matplotlib.pyplot as plt
import numpy as np

def policy_evaluation(problem, policy, gamma=0.9):
    V = np.zeros((problem.state_space[-1][0]+1,problem.state_space[-1][1]+1))
    theta = .0001
    delta = 1
    while delta > theta:
        delta = 0
        for s in problem.state_space:
            old_v = V[s[0],s[1]]
            V[s[0],s[1]] = sum(policy[(s[0],s[1],a)] * problem.expected_return(V, s, a, gamma) for a in problem.action_space)
            delta = max(delta,abs(old_v -V[s[0],s[1]]))
    
    return V

def value_iteration(problem, gamma=0.9):
    V = np.zeros((problem.state_space[-1][0]+1,problem.state_space[-1][1]+1))
    theta = .0001
    delta = 1
    
    while delta > theta:
        delta = 0
        for s in problem.state_space:
            old_v = V[s[0],s[1]]
            V[s[0],s[1]] = max(problem.expected_return(V, s, a, gamma) for a in problem.action_space)
            delta = max(delta,abs(old_v -V[s[0],s[1]]))
    
    pi_star = {
        (x, y, a): 0 for x in range(problem.state_space[0][0],problem.state_space[-1][0]+1) \
        for y in range(problem.state_space[0][1],problem.state_space[-1][1]+1) \
        for a in problem.action_space
    }

    for x,y in problem.state_space:
        # add min of action_space to account for shift in index
        best_a = np.argmax(list(problem.expected_return(V, (x,y), a, gamma) for a in problem.action_space)) + min(problem.action_space)
        pi_star[(x,y,best_a)] = 1
    
    return V, pi_star
    
def policy_iteration(problem, policy, gamma=0.9):
    V = np.zeros((problem.state_space[-1][0]+1,problem.state_space[-1][1]+1))
    new_policy = policy
    old_policy = {}
    
    while old_policy != new_policy:
        old_policy = new_policy
        V = policy_evaluation(problem,old_policy)
        new_policy = {
            (x, y, a): 0 for x in range(problem.state_space[0][0],problem.state_space[-1][0]+1) \
            for y in range(problem.state_space[0][1],problem.state_space[-1][1]+1) \
            for a in problem.action_space
        }
        for x,y in problem.state_space:
            best_a = np.argmax(list(problem.expected_return(V, (x,y), a, gamma) for a in problem.action_space)) + min(problem.action_space)
            new_policy[(x,y,best_a)] = 1
            
        #plt.contour(V)
        #plt.show()
        
    return V, new_policy

def main():
    
    jacksworld = JacksCarRental()
    jacksworld.precompute_transitions()
    random_policy = {
        (x, y, a): .1 for x in range(0, 21) for y in range(0, 21) for a in range(-5, 6)
    }
    
    v_star, pi_star = policy_iteration(jacksworld,random_policy)
    # create graphic representation
    pi_rep = np.zeros((jacksworld.state_space[-1][0]+1,jacksworld.state_space[-1][1]+1))
    rows,cols = pi_rep.shape
    for x in range(rows):
        for y in range(cols):
            pi_rep[x,y] = np.argmax(list(pi_star[(x,y,a)] for a in jacksworld.action_space)) + min(jacksworld.action_space)
    
    plt.contour(v_star)
    plt.title("Jack's Cars Value Function")
    plt.xlabel("Cars at location 2")
    plt.ylabel("Cars at location 1")
    plt.show()
    
    print("Jack's world policy interation")
    print(np.round(v_star,1))
    print(pi_rep)
    
    
    jacksworld = JacksCarRental(modified=True)
    jacksworld.precompute_transitions()
    random_policy = {
        (x, y, a): .1 for x in range(0, 21) for y in range(0, 21) for a in range(-5, 6)
    }
    
    v_star, pi_star = policy_iteration(jacksworld,random_policy)
    # create graphic representation
    pi_rep = np.zeros((jacksworld.state_space[-1][0]+1,jacksworld.state_space[-1][1]+1))
    rows,cols = pi_rep.shape
    for x in range(rows):
        for y in range(cols):
            pi_rep[x,y] = np.argmax(list(pi_star[(x,y,a)] for a in jacksworld.action_space)) + min(jacksworld.action_space)
    
    plt.contour(v_star)
    plt.title("Jack's Cars Value Function - Modified")
    plt.xlabel("Cars at location 2")
    plt.ylabel("Cars at location 1")
    plt.show()
    
    print("Jack's world policy interation - modified")
    print(np.round(v_star,1))
    print(pi_rep)
    
if __name__ == "__main__":
    main()