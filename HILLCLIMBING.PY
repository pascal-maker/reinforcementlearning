import gymnasium as gym
import numpy as np

def computeAction(state, weights):
    """Compute action based on state and weights using linear combination"""
    p = np.matmul(state, weights)
    if p < 0:
        action = 0
    else:
        action = 1
    return action

# Create environment
env = gym.make("CartPole-v1")

# Hill climbing parameters
nr_of_iterations = 2000
nr_of_episodes_per_iteration = 20
noise_scaling = 1e-2  # Increased from 1e-3 for better exploration

# Initialize with random weights
best_weights = 2 * np.random.rand(4) - 1.0
best_reward = 0

for iteration in range(nr_of_iterations):
    # Add noise to current best weights to explore nearby solutions
    weights = best_weights + noise_scaling * (2 * np.random.rand(4) - 1.0)
    total_reward = 0.0
    
    # Test these weights on multiple episodes
    for episode in range(nr_of_episodes_per_iteration):
        state, info = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            action = computeAction(state, weights)
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        total_reward += episode_reward
    
    # Calculate average reward for these weights
    avg_reward = total_reward / nr_of_episodes_per_iteration
    
    # Hill climbing: only accept if better (greedy)
    if avg_reward > best_reward:
        best_reward = avg_reward
        best_weights = weights
        print(f"Iteration {iteration}: New best average reward = {avg_reward:.2f}")

env.close()

print(f"\nBest weights found: {best_weights}")
print(f"Best average reward: {best_reward:.2f}")