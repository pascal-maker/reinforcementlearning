"""
Simulated Annealing Algorithm for CartPole-v1
==============================================
This script uses simulated annealing to find optimal weights for a linear policy.
The algorithm is inspired by the physical process of annealing in metallurgy, where
controlled cooling allows atoms to find low-energy configurations.

Key Concept: Unlike hill climbing (which only accepts better solutions), simulated
annealing sometimes accepts worse solutions. This helps escape local optima.
"""

import gymnasium as gym
import numpy as np

def computeAction(state, weights):
    """
    Compute action based on state and weights using linear combination.
    
    Args:
        state: Array of 4 values [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
        weights: Array of 4 weights to multiply with state features
    
    Returns:
        action: 0 (push left) if weighted sum is negative, 1 (push right) otherwise
    """
    p = np.matmul(state, weights)  # Linear combination: w1*x1 + w2*x2 + w3*x3 + w4*x4
    if p < 0:
        action = 0  # Push cart to the left
    else:
        action = 1  # Push cart to the right
    return action

# Create the CartPole environment
env = gym.make("CartPole-v1")

# ========== Simulated Annealing Parameters ==========
nr_of_iterations = 2000  # Total number of optimization iterations
nr_of_episodes_per_iteration = 20  # Episodes to average over when evaluating weights

# Temperature controls exploration vs exploitation
temperature = 100000  # High initial temperature allows accepting many worse solutions (exploration)
cooling_rate = 0.993  # Multiply temperature by this each iteration (gradual cooling)
                      # After 2000 iterations: T_final ≈ 100000 * 0.993^2000 ≈ 0.08

spread = 0.001  # Standard deviation of Gaussian noise added to weights (small local search)

# Initialize with random weights in range [-1, 1]
best_weights = 2 * np.random.rand(4) - 1.0  # Random starting point
best_reward = 0  # Track the best average reward found so far

# ========== Main Simulated Annealing Loop ==========
for iteration in range(nr_of_iterations):
    # Generate candidate solution by perturbing current best weights
    # Small Gaussian noise creates a neighbor solution (local search)
    current_weights = best_weights + np.random.normal(loc=0, scale=spread, size=4)
    current_reward = 0.0
    
    # Evaluate the candidate weights over multiple episodes to reduce variance
    for episode in range(nr_of_episodes_per_iteration):
        state, info = env.reset()  # Reset environment to initial state
        done = False
        episode_reward = 0.0
        
        # Run one episode until termination
        while not done:
            action = computeAction(state, current_weights)  # Use linear policy
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward  # Accumulate reward (1 per timestep survived)
            done = terminated or truncated  # Episode ends if pole falls or time limit reached
        
        current_reward += episode_reward
    
    # Calculate average reward across all test episodes
    current_reward = current_reward / nr_of_episodes_per_iteration
    
    # ========== Simulated Annealing Acceptance Criterion ==========
    if current_reward > best_reward:
        # Case 1: Better solution - always accept (greedy improvement)
        best_reward = current_reward
        best_weights = current_weights
        print(f"Iteration {iteration}: New best average reward = {current_reward:.2f}, Temp = {temperature:.2f}")
    else:
        # Case 2: Worse solution - accept with probability based on temperature
        # This is the key feature that differentiates simulated annealing from hill climbing
        
        reward_difference = current_reward - best_reward  # Negative value (worse solution)
        
        # Metropolis acceptance criterion: P(accept) = e^(ΔE/T)
        # - Higher temperature → higher probability of accepting worse solutions
        # - Larger negative difference → lower probability
        # - As T decreases, becomes more greedy (less likely to accept worse solutions)
        acceptance_probability = np.exp(reward_difference / temperature)
        
        if np.random.rand() < acceptance_probability:
            # Accept the worse solution (helps escape local optima)
            best_reward = current_reward
            best_weights = current_weights
            print(f"Iteration {iteration}: Accepted worse solution (reward = {current_reward:.2f}), Temp = {temperature:.2f}")
    
    # Decrease temperature (cooling schedule)
    # Early iterations: high T → explore diverse solutions
    # Later iterations: low T → exploit best regions found
    temperature = temperature * cooling_rate

env.close()

print(f"\nBest weights found: {best_weights}")
print(f"Best average reward: {best_reward:.2f}")