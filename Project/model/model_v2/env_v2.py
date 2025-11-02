import gym
from gym import spaces
import numpy as np
import pandas as pd
path = 'Latest\Project\Final.csv'
class EEGWheelchairEnv(gym.Env):
    def __init__(self, data_file):
        super(EEGWheelchairEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # 0: Stop, 1: Forward, 2: Backward
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        
        # Load dataset
        self.data = pd.read_csv(data_file)
        self.current_step = 0
        self.state = np.zeros(3)  # Placeholder for EEG signal
        self.done = False

    def reset(self):
        self.current_step = 0  # Reset step
        self.done = False
        self.state = self.get_next_state()
        return self.state

    def step(self, action):
        reward = self.compute_reward(action)
        self.state = self.get_next_state()
        self.done = self.check_done()
        return self.state, reward, self.done, {}

    def compute_reward(self, action):
        # Customize the reward function based on specific goals
        if action == 1:  # Forward
            return 1.0
        elif action == 2:  # Backward
            return -1.0
        else:  # Stop
            return -0.1

    def get_next_state(self):
        # Retrieve the EEG value and corresponding action from the dataset
        if self.current_step < len(self.data):
            eeg_value = self.data['Value'].iloc[self.current_step]
            action = self.data['Result'].iloc[self.current_step]

            # Update state based on EEG value and action
            self.state = np.array([eeg_value, action == 'Forward', action == 'Backward'])  # One-hot encoding for actions
            self.current_step += 1
            return self.state
        else:
            self.done = True
            return self.state  # Return the last state if done

    def check_done(self):
        return self.done

# Create an instance of the environment with the dataset
env_v2 = EEGWheelchairEnv(data_file=path)  # Replace with your CSV file path
