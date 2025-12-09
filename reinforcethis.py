import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import gymnasium as gym

def get_discounted_rewards(rewards, gamma=0.99):
    """
    Calculates and normalizes discounted rewards (returns) for an episode.
    
    Args:
        rewards: List of immediate rewards received at each time step.
        gamma: Discount factor (0 to 1), weighting future rewards.
        
    Returns:
        Standardized discounted rewards (mean 0, std 1) to stabilize training.
    """
    discounted_rewards = []
    cumulative_total_return = 0
    
    # Iterate backwards to efficiently calculate returns: G_t = r_t + gamma * G_{t+1}
    # We do this backwards because the current return depends on the future return.
    for reward in rewards[::-1]:
        cumulative_total_return = reward + gamma * cumulative_total_return
        discounted_rewards.append(cumulative_total_return)
        
    # Reverse to get correct time order (t=0, t=1, ...)
    discounted_rewards.reverse()
    discounted_rewards = np.array(discounted_rewards)
    
    # Normalize rewards to reduce variance (Standardization)
    # This keeps gradients in a stable range, making training more robust.
    mean_rewards = np.mean(discounted_rewards)
    std_rewards = np.std(discounted_rewards)
    normalized_discounted_rewards = (discounted_rewards - mean_rewards) / (std_rewards + 1e-8)
    
    return normalized_discounted_rewards

class PolicyGradientAgent:
    def __init__(self, state_shape, action_shape, learning_rate=0.001):
        """
        Initializes the agent with a neural network policy.
        """
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.model = self.build_model(learning_rate)

    def build_model(self, learning_rate):
        """
        Builds a simple Feed-Forward Neural Network for the policy.
        Input: State vector
        Output: Probability distribution over actions (Softmax)
        """
        model = Sequential([
            Dense(24, input_shape=self.state_shape, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_shape, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate))
        return model

    def get_action(self, state):
        """
        Selects an action based on the policy's probability distribution.
        """
        # Ensure state has batch dimension: (1, state_dim)
        if state.ndim == 1:
            state = state.reshape([1, state.shape[0]])
            
        # Get action probabilities from the model (Forward pass)
        action_probs = self.model.predict(state, verbose=0).flatten()
        
        # Sample an action stochastically 
        # (exploration vs exploitation is implicitly handled by the probability distribution)
        action = np.random.choice(self.action_shape, p=action_probs)
        return action, action_probs

    def update_policy(self, states, action_probs_history, actions_onehot_history, discounted_rewards):
        """
        Updates the policy using the REINFORCE algorithm approach with a "Train-on-Batch" trick.
        
        Logic:
           We want to increase the probability of actions that led to high rewards (High Advantage)
           and decrease probability of actions that led to low rewards.
           
           Gradient direction = (Actual_Action - Predicted_Prob) * Advantage
           
           We construct a "pseudo-target" (y_train) that represents where we want the probabilities to move:
           y_train = Current_Probs + alpha * Gradient
           
        Args:
            states: History of states in the episode.
            action_probs_history: History of predicted probabilities for those states.
            actions_onehot_history: History of actual actions taken (one-hot encoded).
            discounted_rewards: The 'Advantage' or return for each step.
        """
        # Stack lists into numpy arrays for efficient batch processing
        states = np.vstack(states)
        action_probs = np.vstack(action_probs_history)
        actions_one_hot = np.vstack(actions_onehot_history)
        discounted_rewards = np.vstack(discounted_rewards)

        # Calculate the "Gradient" or direction to shift probabilities
        # (Action_Taken - Current_Prob) gives the direction towards certainty for that action.
        # Multiplying by 'discounted_rewards' scales this:
        #   - Positive Return: Push probability TOWARDS 1.0
        #   - Negative Return: Push probability AWAY (towards 0.0)
        gradients = (actions_one_hot - action_probs) * discounted_rewards
        
        # Define a scaling factor (alpha)
        # This acts similarly to a learning rate modifier for the target construction.
        # Since we use an optimizer (Adam) later, 1.0 is a reasonable default here.
        alpha = 1.0 
        
        # Construct the target labels (pseudo-labels)
        # We tell the model: "For this input, you SHOULD have predicted this distribution"
        y_train = action_probs + (alpha * gradients)
        
        # Train the model on this batch of states and targets
        history = self.model.train_on_batch(states, y_train)
        return history

def train(episodes=100):
    """
    Main training loop.
    """
    # Initialize environment and agent
    env = gym.make('CartPole-v1')
    state_shape = env.observation_space.shape
    action_shape = env.action_space.n
    agent = PolicyGradientAgent(state_shape, action_shape)

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        
        # Storage for current episode data
        episode_states = []
        episode_rewards = []
        episode_action_probs = []
        episode_actions_onehot = []

        # Play one full episode (Monte Carlo)
        while not done:
            # 1. Select action
            action, action_probs = agent.get_action(state)
            
            # 2. Store execution data for training
            episode_states.append(state)
            episode_action_probs.append(action_probs)
            
            # Encode action as one-hot vector (e.g., [0, 1] if action is 1)
            one_hot_action = np.zeros(action_shape)
            one_hot_action[action] = 1
            episode_actions_onehot.append(one_hot_action)
            
            # 3. Step in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_rewards.append(reward)
            state = next_state

        # 4. Calculate returns (discounted rewards) from the episode
        discounted_rewards = get_discounted_rewards(episode_rewards)
        
        # 5. Update policy network
        loss = agent.update_policy(
            episode_states, 
            episode_action_probs, 
            episode_actions_onehot, 
            discounted_rewards
        )
        
        print(f"Episode: {episode+1}, Total Reward: {sum(episode_rewards)}, Loss: {loss}")

if __name__ == "__main__":
    # Train for a few episodes to demonstrate functionality
    train(episodes=5)
