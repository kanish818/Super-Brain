import numpy as np
import tensorflow as tf
from rlmodel import DQNAgent
from env import EEGWheelchairEnv

def load_model(weights_path, state_size, action_size):
    agent = DQNAgent(state_size, action_size)
    agent.load(weights_path)
    return agent

def predict_action(agent, state):
    state = np.reshape(state, [1, state_size])
    return agent.act(state)

if __name__ == "__main__":
    weights_path = "dqn_model.h5"
    env = EEGWheelchairEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = load_model(weights_path, state_size, action_size)
    
    state = env.reset()
    done = False
    
    while not done:
        action = predict_action(agent, state)
        next_state, reward, done, _ = env.step(action)
        env.render()
        state = next_state
