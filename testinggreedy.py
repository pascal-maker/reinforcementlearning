import random
import numpy as np


def epsilon_greedy(epsilon, possible_actions, action_values):
    """
    Epsilon-greedy action selection strategy.
    
    Args:
        epsilon: Probability of selecting a random action (exploration)
        possible_actions: List of possible actions to choose from
        action_values: Array or list of estimated values for each action
        
    Returns:
        Selected action (index)
    """
    r = random.random()
    if r <= epsilon:
        # Exploration: choose random action
        action = random.choice(possible_actions)
    else:
        # Exploitation: choose action with highest value
        action = np.argmax(action_values)
    return action


# Example usage as a class method (for integration with bandit/RL agents)
class EpsilonGreedyAgent:
    def __init__(self, n_actions, epsilon=0.1):
        """
        Initialize epsilon-greedy agent.
        
        Args:
            n_actions: Number of possible actions
            epsilon: Exploration rate (default 0.1)
        """
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.action_values = np.zeros(n_actions)
        self.action_counts = np.zeros(n_actions)
        
    def select_action(self):
        """Select action using epsilon-greedy strategy."""
        possible_actions = list(range(self.n_actions))
        return epsilon_greedy(self.epsilon, possible_actions, self.action_values)
    
    def update(self, action, reward):
        """Update action value estimates based on received reward."""
        self.action_counts[action] += 1
        # Incremental mean update formula: new_mean = old_mean + (new_reward - old_mean) / n
        # This computes the running average without storing all past rewards (memory-efficient)
        # - (reward - self.action_values[action]): error between new reward and current estimate
        # - / self.action_counts[action]: scale error by number of observations
        # - +=: add scaled error to update our estimate
        # Equivalent to: mean = sum(all_rewards) / n, but computed incrementally
        self.action_values[action] += (reward - self.action_values[action]) / self.action_counts[action]


# Example usage
if __name__ == "__main__":
    # Test the standalone function
    epsilon = 0.1
    possible_actions = [0, 1, 2, 3]
    action_values = [0.5, 0.8, 0.3, 0.9]
    
    print("Standalone function example:")
    for i in range(10):
        action = epsilon_greedy(epsilon, possible_actions, action_values)
        print(f"Step {i+1}: Selected action {action}")
    
    print("\n" + "="*50 + "\n")
    
    # Test the agent class
    print("Agent class example:")
    agent = EpsilonGreedyAgent(n_actions=4, epsilon=0.1)
    
    # Simulate some interactions
    for i in range(10):
        action = agent.select_action()
        # Simulate reward (random for demo)
        reward = random.random()
        agent.update(action, reward)
        print(f"Step {i+1}: Action {action}, Reward {reward:.3f}")
    
    print(f"\nFinal action values: {agent.action_values}")
    print(f"Action counts: {agent.action_counts}")


