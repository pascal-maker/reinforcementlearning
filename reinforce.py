import numpy as np
import gymnasium as gym
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create environment
env = gym.make('CartPole-v1')
action_shape = env.action_space.n
state_shape = env.observation_space.shape

# Define model
def build_model():
    model = Sequential([
        Dense(24, input_shape=state_shape, activation='relu'),
        Dense(24, activation='relu'),
        Dense(action_shape, activation='softmax')
    ])
    return model

model = build_model()

def get_action(state):
    # Fix: Ensure state is correct shape (1, state_dim)
    if state.ndim == 1:
        state = state.reshape([1, state.shape[0]])
    
    # Predict probabilities
    action_probability_distribution = model.predict(state, verbose=0).flatten()
    
    # Sample action
    action = np.random.choice(action_shape, p=action_probability_distribution)
    
    return action, action_probability_distribution

# Test the function
if __name__ == "__main__":
    try:
        state, _ = env.reset()
        action, probs = get_action(state)
        print(f"State: {state}")
        print(f"Action: {action}")
        print(f"Probabilities: {probs}")
        print("Script fixed and ran successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")