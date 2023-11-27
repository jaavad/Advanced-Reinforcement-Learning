import random
from enum import IntEnum
from typing import Tuple, Optional, List
from gym import Env, spaces
from gym.utils import seeding
from gym.envs.registration import register


def register_env() -> None:
    """Register custom gym environment so that we can use `gym.make()`

    In your main file, call this function before using `gym.make()` to use the Four Rooms environment.
        register_env()
        env = gym.make('WindyGridWorld-v0')

    """
    # TODO
    register(id="WindyGridWorld-v0", entry_point="env:WindyGridWorldEnv")
    
    register(id="WindyGridWorldKings-v0", entry_point="env:WindyGridWorldKings")
    register(id="WindyGridWorldKings-v1", entry_point="env:WindyGridWorldKings", kwargs={'stay': True})
    
    register(id="WindyGridWorldStochastic-v0", entry_point="env:WindyGridWorldStochastic")
    register(id="WindyGridWorldStochastic-v1", entry_point="env:WindyGridWorldStochastic", kwargs={'stay': True})

class Action(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


def actions_to_dxdy(action: Action) -> Tuple[int, int]:
    """
    Helper function to map action to changes in x and y coordinates
    Args:
        action (Action): taken action
    Returns:
        dxdy (Tuple[int, int]): Change in x and y coordinates
    """
    mapping = {
        Action.LEFT: (-1, 0),
        Action.DOWN: (0, -1),
        Action.RIGHT: (1, 0),
        Action.UP: (0, 1),
    }
    return mapping[action]


class WindyGridWorldEnv(Env):
    def __init__(self):
        """Windy grid world gym environment
        """

        # Grid dimensions (x, y)
        self.rows = 10
        self.cols = 7

        # Wind
        self.wind = {}
        for x in range(self.rows):
            for y in range(self.cols):
                # 3,4,5,8 = up by 1, 6,7 = up by 2, 0,1,2,9 = up by 0
                if x < 3 or x == 9:
                    self.wind[(x,y)] = 0
                elif x == 6 or x == 7: 
                    self.wind[(x,y)] = 2
                else:
                    self.wind[(x,y)] = 1

        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(self.rows), spaces.Discrete(self.cols))
        )

        # Set start_pos and goal_pos
        self.start_pos = (0, 3)
        self.goal_pos = (7, 3)
        self.agent_pos = None

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Fix seed of environment

        In order to make the environment completely reproducible, call this function and seed the action space as well.
            env = gym.make(...)
            env.seed(seed)
            env.action_space.seed(seed)

        This function does not need to be used for this assignment, it is given only for reference.
        """

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action: Action) -> Tuple[Tuple[int, int], float, bool, dict]:
        """Take one step in the environment.

        Takes in an action and returns the (next state, reward, done, info).
        See https://github.com/openai/gym/blob/master/gym/core.py#L42-L58 foand r more info.

        Args:
            action (Action): an action provided by the agent

        Returns:
            observation (object): agent's observation after taking one step in environment (this would be the next state s')
            reward (float) : reward for this transition
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning). Not used in this assignment.
        """

        reward = None
        done = None


        action_taken = actions_to_dxdy(action)
        
        # move and check for boundaries
        tentative_next_state = tuple(map(sum, zip(self.agent_pos, action_taken)))
        if tentative_next_state[0] >= self.rows or tentative_next_state[1] >= self.cols or tentative_next_state[0] < 0 or tentative_next_state[1] < 0:
            next_pos = self.agent_pos
        else:
            next_pos = tentative_next_state
            
        # add effect of wind
        wind = self.wind[(self.agent_pos)]
        next_y_wind = next_pos[1]
        for i in range(wind): # attempt moving up 0,1,2 times depending on wind
            if next_y_wind < self.cols - 1: # we can move up
                next_y_wind += 1
        
        self.agent_pos = (next_pos[0],next_y_wind)
        
        # Check if goal was reached
        if self.agent_pos == self.goal_pos:
            done = True
            reward = -1.0
        else:
            done = False
            reward = -1.0
        

        return self.agent_pos, reward, done, {}
        

class KingAction(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    UP_LEFT = 4
    UP_RIGHT = 5
    DOWN_LEFT = 6
    DOWN_RIGHT = 7
    STAY = 8 # don't let this get picked if not stay


def king_actions_to_dxdy(action: KingAction) -> Tuple[int, int]:
    """
     function to map action to changes in x and y coordinates

    """
    mapping = {
        KingAction.LEFT: (-1, 0),
        KingAction.DOWN: (0, -1),
        KingAction.RIGHT: (1, 0),
        KingAction.UP: (0, 1),
        KingAction.UP_LEFT: (-1, 1),
        KingAction.UP_RIGHT: (1, 1),
        KingAction.DOWN_LEFT: (-1, -1),
        KingAction.DOWN_RIGHT: (1, -1),
        KingAction.STAY: (0, 0)
    }
    return mapping[action]
        
class WindyGridWorldKings(Env):
    def __init__(self, stay=False): # stay = is staying a valid action?

        self.stay = stay

        # Grid dimensions (x, y)
        self.rows = 10
        self.cols = 7

        # Wind
        self.wind = {}
        for x in range(self.rows):
            for y in range(self.cols):
                # 3,4,5,8 = up by 1, 6,7 = up by 2, 0,1,2,9 = up by 0
                if x < 3 or x == 9:
                    self.wind[(x,y)] = 0
                elif x == 6 or x == 7: 
                    self.wind[(x,y)] = 2
                else:
                    self.wind[(x,y)] = 1
                    
        self.action_space = spaces.Discrete(len(KingAction) - 1 + self.stay) # can't pick "stay" unless enabled by args
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(self.rows), spaces.Discrete(self.cols))
        )

        # Set start_pos and goal_pos
        self.start_pos = (0, 3)
        self.goal_pos = (7, 3)
        self.agent_pos = None

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Fix seed of environment

        In order to make the environment completely reproducible, call this function and seed the action space as well.
            env = gym.make(...)
            env.seed(seed)
            env.action_space.seed(seed)

        This function does not need to be used for this assignment, it is given only for reference.
        """

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action: KingAction) -> Tuple[Tuple[int, int], float, bool, dict]:
        """Take one step in the environment.

        Takes in an action and returns the (next state, reward, done, info).
        See https://github.com/openai/gym/blob/master/gym/core.py#L42-L58 foand r more info.

        Args:
            action (Action): an action provided by the agent

        Returns:
            observation (object): agent's observation after taking one step in environment (this would be the next state s')
            reward (float) : reward for this transition
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning). Not used in this assignment.
        """

        reward = None
        done = None
        

        action_taken = king_actions_to_dxdy(action)
        
        # move and check for boundaries
        tentative_next_state = tuple(map(sum, zip(self.agent_pos, action_taken)))
        if tentative_next_state[0] >= self.rows or tentative_next_state[1] >= self.cols or tentative_next_state[0] < 0 or tentative_next_state[1] < 0:
            next_pos = self.agent_pos
        else:
            next_pos = tentative_next_state
            
        # add effect of wind
        wind = self.wind[(self.agent_pos)]
        next_y_wind = next_pos[1]
        for i in range(wind): # attempt moving up 0,1,2 times depending on wind
            if next_y_wind < self.cols - 1: # we can move up
                next_y_wind += 1
        
        self.agent_pos = (next_pos[0],next_y_wind)
        
        # Check if goal was reached
        if self.agent_pos == self.goal_pos:
            done = True
            reward = -1.0
        else:
            done = False
            reward = -1.0
        

        return self.agent_pos, reward, done, {}
        
class WindyGridWorldStochastic(Env):
    def __init__(self, stay=False): # stay = is staying a valid action?
        """Windy grid world gym environment
        """
        
        self.stay = stay

        # Grid dimensions (x, y)
        self.rows = 10
        self.cols = 7

        # Wind
        self.wind = {}
        for x in range(self.rows):
            for y in range(self.cols):
                # 3,4,5,8 = up by 1, 6,7 = up by 2, 0,1,2,9 = up by 0
                if x < 3 or x == 9:
                    self.wind[(x,y)] = 0
                elif x == 6 or x == 7: 
                    self.wind[(x,y)] = 2
                else:
                    self.wind[(x,y)] = 1
                    
        self.action_space = spaces.Discrete(len(KingAction) - 1 + self.stay) # can't pick "stay" unless enabled by args
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(self.rows), spaces.Discrete(self.cols))
        )

        # Set start_pos and goal_pos
        self.start_pos = (0, 3)
        self.goal_pos = (7, 3)
        self.agent_pos = None

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Fix seed of environment

        In order to make the environment completely reproducible, call this function and seed the action space as well.
            env = gym.make(...)
            env.seed(seed)
            env.action_space.seed(seed)

        This function does not need to be used for this assignment, it is given only for reference.
        """

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action: KingAction) -> Tuple[Tuple[int, int], float, bool, dict]:
        """Take one step in the environment.

        Takes in an action and returns the (next state, reward, done, info).
        See https://github.com/openai/gym/blob/master/gym/core.py#L42-L58 foand r more info.

        Args:
            action (Action): an action provided by the agent

        Returns:
            observation (object): agent's observation after taking one step in environment (this would be the next state s')
            reward (float) : reward for this transition
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning). Not used in this assignment.
        """

        reward = None
        done = None
        

        action_taken = king_actions_to_dxdy(action)
        
        # move and check for boundaries
        tentative_next_state = tuple(map(sum, zip(self.agent_pos, action_taken)))
        if tentative_next_state[0] >= self.rows or tentative_next_state[1] >= self.cols or tentative_next_state[0] < 0 or tentative_next_state[1] < 0:
            next_pos = self.agent_pos
        else:
            next_pos = tentative_next_state
            
        # add effect of wind
        wind = self.wind[(self.agent_pos)]
        # make it stochastic
        wind_change = random.random()
        if wind != 0: # change effect of wind
            if wind_change < 0.33333:
                wind -= 1
            elif wind_change < 0.66667:
                wind += 1
        
        next_y_wind = next_pos[1]
        for i in range(wind): # attempt moving up 0,1,2 times depending on wind
            if next_y_wind < self.cols - 1: # we can move up
                next_y_wind += 1
        
        self.agent_pos = (next_pos[0],next_y_wind)
        
        # Check if goal was reached
        if self.agent_pos == self.goal_pos:
            done = True
            reward = -1.0
        else:
            done = False
            reward = -1.0
        

        return self.agent_pos, reward, done, {}
