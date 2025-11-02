import gym
from gym import spaces
import numpy as np

class EEGWheelchairEnv(gym.Env):
    def __init__(self):
        super(EEGWheelchairEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # 0: backward, 1: forward
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.state = None
        self.done = False
    
    def reset(self):
        self.state = np.zeros(3)
        self.done = False
        return self.state
    
    def step(self, action):
        reward = self.compute_reward(action)
        self.state = self.get_next_state()
        self.done = self.check_done()
        return self.state, reward, self.done, {}
    
    def compute_reward(self, action):
        if action == 1:  # Forward
            reward = 1.0
        elif action==-1:  # Backward
            reward = -1.0
        else:
            reward=0

        return reward
    
    def get_next_state(self):
        # changes needed to be made 
        return np.random.rand(3)
    
    
    def check_done(self):
        return np.random.rand() > 0.95

env = EEGWheelchairEnv()
