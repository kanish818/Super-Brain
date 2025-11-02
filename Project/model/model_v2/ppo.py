import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import gym
from env_v2 import EEGWheelchairEnv  # Import the environment
path = 'Latest\Project\Final.csv'
class PPOAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99  
        self.epsilon = 0.2  
        self.learning_rate = 0.001
        
        # Actor and Critic models
        self.actor = self.build_actor()
        self.critic = self.build_critic()

    def build_actor(self):
        input_layer = Input(shape=(self.state_size,))
        hidden_layer = Dense(64, activation='relu')(input_layer)
        hidden_layer = Dense(64, activation='relu')(hidden_layer)
        output_layer = Dense(self.action_size, activation='softmax')(hidden_layer)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=self.ppo_loss)
        return model

    def build_critic(self):
        input_layer = Input(shape=(self.state_size,))
        hidden_layer = Dense(64, activation='relu')(input_layer)
        hidden_layer = Dense(64, activation='relu')(hidden_layer)
        output_layer = Dense(1, activation='linear')(hidden_layer)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def ppo_loss(self, y_true, y_pred):
        advantages, actions, old_prediction = y_true[:, :1], y_true[:, 1:1 + self.action_size], y_true[:, 1 + self.action_size:]
        
        prob = tf.reduce_sum(actions * y_pred, axis=1)
        old_prob = tf.reduce_sum(actions * old_prediction, axis=1)
        ratio = tf.exp(tf.math.log(prob) - tf.math.log(old_prob))
        
        p1 = ratio * advantages
        p2 = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        loss = -tf.reduce_mean(tf.minimum(p1, p2))
        return loss

    def act(self, state):
        probabilities = self.actor.predict(state)[0]
        action = np.random.choice(self.action_size, p=probabilities)
        return action

    def train(self, states, actions, advantages, returns, old_predictions):
        y_true = np.hstack([advantages, actions, old_predictions])
        self.actor.fit(states, y_true, epochs=1, verbose=0)
        self.critic.fit(states, returns, epochs=1, verbose=0)

    def discount_rewards(self, rewards, dones):
        discounted_rewards = []
        cumulative = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                cumulative = 0
            cumulative = reward + self.gamma * cumulative
            discounted_rewards.insert(0, cumulative)
        return discounted_rewards


state_size = 3  # Adjust this if the state size changes
action_size = 3
agent = PPOAgent(state_size, action_size)

# Create an instance of the environment with the dataset
env = EEGWheelchairEnv(data_file=path)  # Replace with your CSV file path

episodes = 100
batch_size = 64
for episode in range(episodes):
    state = env.reset()  # Reset the environment
    states, actions, rewards, old_predictions, dones = [], [], [], [], []
    done = False
    while not done:
        state = np.reshape(state, [1, state_size])
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)

        action_onehot = np.zeros(action_size)
        action_onehot[action] = 1

        states.append(state)
        actions.append(action_onehot)
        rewards.append(reward)
        dones.append(done)

        state = next_state

    returns = agent.discount_rewards(rewards, dones)
    advantages = np.array(returns) - agent.critic.predict(np.vstack(states))

    old_predictions = agent.actor.predict(np.vstack(states))
    agent.train(np.vstack(states), np.vstack(actions), advantages, np.vstack(returns), old_predictions)

    if episode % 10 == 0:
        print(f"Episode {episode} complete")

# Save the models
agent.actor.save('ppo_wheelchair_actor.h5')
agent.critic.save('ppo_wheelchair_critic.h5')
