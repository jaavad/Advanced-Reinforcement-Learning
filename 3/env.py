from scipy.stats import poisson
import numpy as np
from enum import IntEnum
from typing import Tuple


class Action(IntEnum):
 
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


def actions_to_dxdy(action: Action) -> Tuple[int, int]:
    
    ## function to map action to changes in x and y coordinates

    mapping = {
        Action.LEFT: (-1, 0),
        Action.DOWN: (0, -1),
        Action.RIGHT: (1, 0),
        Action.UP: (0, 1),
    }
    return mapping[action]


class Gridworld5x5:
 
    def __init__(self) -> None:
 
        self.rows = 5
        self.cols = 5
        self.state_space = [
            (x, y) for x in range(0, self.rows) for y in range(0, self.cols)
        ]
        self.action_space = list(Action)

        # setting the locations of A and B, the next locations, and their rewards
        self.A = (1,4)
        self.A_prime = (1,0)
        self.A_reward = 10
        self.B = (3,4)
        self.B_prime = (3,2)
        self.B_reward = 5

    def transitions(
        self, state: Tuple, action: Action
    ) -> Tuple[Tuple[int, int], float]:
        ##Get transitions from given (state, action) pair.

        
        # checking if current state is A and B and return the next state and corresponding reward
        if state == self.A:
            next_state = self.A_prime
            reward = self.A_reward
        elif state == self.B:
            next_state = self.B_prime
            reward = self.B_reward
        else: # Else, checking if the next step is within boundaries and return next state and reward
            action_tuple = actions_to_dxdy(action)
            tentative_next_state = tuple(map(sum, zip(state, action_tuple)))
            if tentative_next_state[0] > 4 or tentative_next_state[1] > 4 or tentative_next_state[0] < 0 or tentative_next_state[1] < 0:
                next_state = state
                reward = -1
            else: # standard in bounds move
                next_state = tentative_next_state
                reward = 0
        
        return next_state, reward

    def expected_return(
        self, V, state: Tuple[int, int], action: Action, gamma: float
    ) -> float:
        ##Compute the expected_return for all transitions from the (s,a) pair

        next_state, reward = self.transitions(state, action)
        # computing the expected return
        # deterministic transitions make this easy
        ret = reward + gamma * V[next_state[0],next_state[1]]

        return ret


class JacksCarRental:
    def __init__(self, modified: bool = False) -> None:
        """
        Args:
           modified : False = original problem Q6a, True = modified problem for Q6b

        State: # cars at location A, # cars at location B

        Action (int): -5 to +5
            Positive if moving cars from location A to B
            Negative if moving cars from location B to A
        """
        self.modified = modified

        self.action_space = list(range(-5, 6))

        self.rent_reward = 10
        self.move_cost = 2

        # For modified problem
        self.overflow_cars = 10
        self.overflow_cost = 4

        # Rent and return Poisson process parameters
        # Save as an array for each location (Loc A, Loc B)
        self.rent = [poisson(3), poisson(4)]
        self.return_ = [poisson(3), poisson(2)]

        # Max number of cars at end of day
        self.max_cars_end = 20
        # Max number of cars at start of day
        self.max_cars_start = self.max_cars_end + max(self.action_space)

        self.state_space = [
            (x, y)
            for x in range(0, self.max_cars_end + 1)
            for y in range(0, self.max_cars_end + 1)
        ]

        # Store all possible transitions here as a multi-dimensional array (locA, locB, action, locA', locB')
        # This is the 3-argument transition function p(s'|s,a)
        self.t = np.zeros(
            (
                self.max_cars_end + 1,
                self.max_cars_end + 1,
                len(self.action_space),
                self.max_cars_end + 1,
                self.max_cars_end + 1,
            ),
        )

        # Store all possible rewards (locA, locB, action)
        # This is the reward function r(s,a)
        self.r = np.zeros(
            (self.max_cars_end + 1, self.max_cars_end + 1, len(self.action_space))
        )

    def _open_to_close(self, loc_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Computing the probability of ending the day with s_end \in [0,20] cars given that the location started with s_start \in [0, 20+5] cars.

        Args:
            loc_idx : the location index. 0 is for A and 1 is for B. All other values are invalid
        Returns:
            probs : list of probabilities for all possible combination of s_start and s_end
            rewards : average rewards for all possible s_start
        """
        probs = np.zeros((self.max_cars_start + 1, self.max_cars_end + 1))
        rewards = np.zeros(self.max_cars_start + 1)
        for start in range(probs.shape[0]):
            # Calculating average rewards.
            for k in range(probs.shape[0]):
                # min(start,k) is the actual number of cars rented, but probability still comes from trying to rent "k"
                rewards[start] += self.rent_reward * min(start,k) * self.rent[loc_idx].pmf(k)

            # Calculating probabilities
            # Loop over every possible s_end
            for end in range(probs.shape[1]):
                prob = 0.0

                min_rent = max(0, start - end)

                # Loop over all possible rent scenarios and compute probabilities
                for k in range(min_rent, probs.shape[0]):
                    # some of these are == 1 good, but some are slightly less than 1!
                    prob += self.rent[loc_idx].pmf(min(k,start)) * self.return_[loc_idx].pmf(end-start+k)
                
                probs[start, end] = prob
        
        return probs, rewards

    def _calculate_cost(self, state: Tuple[int, int], action: int) -> float:
        ##function to compute the cost of moving cars for a given (state, action)

        
        if not self.modified:
            cost = self.move_cost * abs(action)
        else:
            if action > 0: #moving A to B, we get 1 move free
                cost = self.move_cost * (abs(action) - 1)
            else:
                cost = self.move_cost * abs(action)
            # then we need to add 4 if either overflows
            a_cars, b_cars = state
            if a_cars - action > self.overflow_cars:
                cost += self.overflow_cost
            if b_cars + action > self.overflow_cars:
                cost += self.overflow_cost
        
        return cost

    def _valid_action(self, state: Tuple[int, int], action: int) -> bool:
        ##function to check if this action is valid for the given state
        
        if state[0] < action or state[1] < -(action):
            return False
        else:
            return True

    def precompute_transitions(self) -> None:
        """Function to precompute the transitions and rewards.

        This function should have been run at least once before calling expected_return().
        You can call this function in __init__() or separately.

        """
        # Calculate open_to_close for each location
        day_probs_A, day_rewards_A = self._open_to_close(0)
        day_probs_B, day_rewards_B = self._open_to_close(1)

        # Perform action first then calculate daytime probabilities
        for locA in range(self.max_cars_end + 1):
            for locB in range(self.max_cars_end + 1):
                for ia, action in enumerate(self.action_space):
                    # Check boundary conditions
                    if not self._valid_action((locA, locB), action):
                        self.t[locA, locB, ia, :, :] = 0
                        self.r[locA, locB, ia] = 0
                    else:
                        # Calculating day rewards from renting
                        # Use day_rewards_A and day_rewards_B and _calculate_cost()
                        cost = self._calculate_cost((locA,locB), action)
                        self.r[locA, locB, ia] = day_rewards_A[locA]+day_rewards_B[locB]-cost

                        # Loop over all combinations of locA_ and locB_
                        for locA_ in range(self.max_cars_end + 1):
                            for locB_ in range(self.max_cars_end + 1):

                                # Calculating transition probabilities
                                # Use the probabilities computed from open_to_close
                                self.t[locA, locB, ia, locA_, locB_] = day_probs_A[locA,locA_]*day_probs_B[locB,locB_]


    def expected_return(
        self, V, state: Tuple[int, int], action: Action, gamma: float
    ) -> float:
        ## Computing the expected_return for all transitions from the (s,a) pair, i.e. do a 1-step Bellman backup.


        next_state_probs = self.transitions(state, action)
        reward = self.rewards(state, action)
        # computeing the expected return
        ret = reward
        for locA_ in range(self.max_cars_end + 1):
            for locB_ in range(self.max_cars_end + 1):
                ret += gamma * next_state_probs[locA_, locB_] * V[locA_, locB_]
        return ret

    def transitions(self, state: Tuple, action: Action) -> np.ndarray:
        ##Get transition probabilities for given (state, action) pair.

        locA, locB = state
        # need to add 5 to convert action to ia (-5 -> 0)
        probs = self.t[locA, locB, action+5]
        return probs

    def rewards(self, state, action) -> float:
        ##Reward function r(s,a)

        locA, locB = state
        return self.r[locA, locB, action+5]
