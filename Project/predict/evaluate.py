import numpy as np
import matplotlib.pyplot as plt
from model.rl_model.env import EEGWheelchairEnv
from model.rl_model.rlmodel import DQNAgent

# Load trained model
model = tf.keras.models.load_model('dqn_wheelchair_model.keras')

# Initialize environment
env = EEGWheelchairEnv()

# Evaluation parameters
episodes = 100
scores = []

for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, 3])  # Adjust based on state size
    total_reward = 0
    
    for time in range(500):  # Run until done or max steps
        action = model.predict(state)
        action = np.argmax(action[0])  # Choose the best action
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 3])
        total_reward += reward
        state = next_state
        
        if done:
            scores.append(total_reward)
            print(f"Episode {e+1}/{episodes} - Score: {total_reward}")
            break

# Plotting the evaluation results
plt.plot(scores)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('Evaluation of Trained DQN Model')
plt.show()

# Statistical Analysis
print(f"Average Score: {np.mean(scores)}")
print(f"Score Standard Deviation: {np.std(scores)}")
