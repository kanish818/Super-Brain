import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, Input
from tensorflow.keras.optimizers import Adam
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
from stable_baselines3.common.policies import MlpLstmPolicy
from stable_baselines3.common.vec_env import VecNormalize
from env_v3 import EEGWheelchairEnv

# Wrap environment for vectorized training
env = DummyVecEnv([lambda: EEGWheelchairEnv()])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Proximal Policy Optimization model with LSTM policy to handle EEG signal time dependence
model = PPO(MlpLstmPolicy, env, verbose=1, learning_rate=0.0003, n_steps=256, batch_size=64, n_epochs=10)

# Training the model
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_eeg_wheelchair_model")

# Load and test the model
model = PPO.load("ppo_eeg_wheelchair_model")

# Test environment
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Done: {done}")
