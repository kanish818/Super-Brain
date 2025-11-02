import gym
from gym import spaces
import numpy as np
import serial  # For receiving data from Arduino
import time

class EEGWheelchairEnv(gym.Env):
    def __init__(self, serial_port='COM3', baudrate=9600, max_steps=1000):
        super(EEGWheelchairEnv, self).__init__()
        
        self.action_space = spaces.Discrete(3)  # 0: backward, 1: forward, 2: stop
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        # Serial connection to receive EEG data from Arduino
        self.serial_connection = serial.Serial(serial_port, baudrate)
        self.state = None
        self.max_steps = max_steps
        self.current_step = 0
        self.done = False
        self.target_position = 5.0  
        self.wheelchair_position = 0.0 
        self.velocity = 0.1  
        self.start_time = time.time()

    def reset(self):
        self.current_step = 0
        self.wheelchair_position = 0.0
        self.state = np.zeros(3) 
        self.done = False
        self.start_time = time.time()
        return self.state
    
    def step(self, action):
        # Apply action to change wheelchair position
        if action == 1:  # Forward
            self.wheelchair_position += self.velocity
        elif action == 0:  # Backward
            self.wheelchair_position -= self.velocity
        elif action == 2:
            self.wheelchair_position = 0

        # Compute reward
        reward = self.compute_reward(action)
        
        # Receive EEG data from Arduino (read signals from serial port)
        self.state = self.receive_eeg_data()
        
        # Check if done
        self.current_step += 1
        self.done = self.check_done()

        return self.state, reward, self.done, {}
    
    def receive_eeg_data(self):
        # Example: read from Arduino's serial port for EEG signal
        try:
            eeg_data = self.serial_connection.readline().decode('utf-8').strip().split(',')
            eeg_values = np.array([float(eeg_data[0]), float(eeg_data[1]), float(eeg_data[2])])  # ref, +ve, -ve
            return eeg_values
        except Exception as e:
            print(f"Error reading EEG data: {e}")
            return np.random.rand(3)  # Return random EEG-like signals in case of error
    
    def compute_reward(self, action):
        distance_to_target = self.target_position - self.wheelchair_position
        reward = 0.0
        
        if action == 1:  # Forward
            reward = 1.0 if distance_to_target > 0 else -1.0  # Reward for moving towards the target
        elif action == 0:  # Backward
            reward = -1.0  # Penalize moving backward
        elif action == 2:  # Stop
            reward = -0.1 if distance_to_target > 0 else 1.0  # Penalize stopping before reaching target
        
        return reward
    
    def check_done(self):
        # End episode if wheelchair reaches target or max steps exceeded
        if abs(self.target_position - self.wheelchair_position) < 0.1:
            return True
        if self.current_step >= self.max_steps:
            return True
        return False

env = EEGWheelchairEnv()
