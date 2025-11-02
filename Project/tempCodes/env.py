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
        self.state = self.get_next_state(action)
        self.done = self.check_done()
        return self.state, reward, self.done, {}
    
    def compute_reward(self, action):
        if action == 1:  # Forward
            reward = 1.0
        elif action == 0:  # Backward
            reward = -1.0
        else:
            reward = 0.0  # No movement

        return reward
    
    def get_next_state(self, action):
        # Simple simulation: the next state is a random small change from the current state, influenced by the action
        noise = np.random.normal(0, 0.1, size=self.state.shape)
        if action == 1:  # Forward
            self.state += np.array([1.0, 0.0, 0.0]) + noise
        elif action == 0:  # Backward
            self.state += np.array([-1.0, 0.0, 0.0]) + noise
        else:  # No movement
            self.state += noise
        return self.state
    
    def check_done(self):
        # End the episode randomly or if the state exceeds a certain limit (for simulation purposes)
        return np.any(np.abs(self.state) > 10) or np.random.rand() > 0.95

env = EEGWheelchairEnv()
