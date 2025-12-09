import os
# Set environment variable to use legacy Keras for compatibility with older TensorFlow versions
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import sys
import tensorflow as tf

# ==============================================================================
# COMPATIBILITY PATCHING (Same as tryingkeras.py)
# ==============================================================================
# Import tf_keras for backward compatibility with TensorFlow 2.x and Keras 2.x
import tf_keras

# Patch 1: Inject __version__ into tf_keras (missing in some versions)
# This ensures the module has a version attribute if it's absent
if not hasattr(tf_keras, '__version__'):
    tf_keras.__version__ = '2.15.0'

# Patch 2: Ensure model_from_config
# Add the model_from_config function if it's missing from tf_keras.models
if not hasattr(tf_keras.models, 'model_from_config'):
     try:
         from tf_keras.saving import model_from_config
         tf_keras.models.model_from_config = model_from_config
     except ImportError:
         pass  # Silently fail if import fails

# Patch 3: Force tensorflow.keras to point to patched tf_keras
# Redirect TensorFlow's Keras module to use the patched tf_keras for consistency
tf.keras = tf_keras
sys.modules['tensorflow.keras'] = tf_keras
sys.modules['tensorflow.keras.models'] = tf_keras.models

# ==============================================================================
# IMPORTS
# ==============================================================================

# Import backend for low-level operations
from tensorflow.keras import backend as K
# Import layers for building the neural network
from tensorflow.keras.layers import Dense, Activation, Input
# Import Model and load_model for creating and loading models
from tensorflow.keras.models import Model, load_model
# Use legacy Adam optimizer for compatibility with older TF versions
from tensorflow.keras.optimizers.legacy import Adam
# NumPy for array operations
import numpy as np

# Import Gymnasium (modern Gym) for the environment
import gymnasium as gym
from gymnasium import wrappers  # Wrappers for environment modification (unused here)

# Disable eager execution to run in graph mode for better performance/stability
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# Suppress TensorFlow logging to reduce verbosity during training
tf.get_logger().setLevel('ERROR')
# tf.config.experimental_run_functions_eagerly(False) # Deprecated and often conflicts with disable_eager_execution

# ==============================================================================
# AGENT CLASS DEFINITION
# ==============================================================================
# This class implements a simple Advantage Actor-Critic (A2C) agent for reinforcement learning.
class Agent(object):
    def __init__(self, alpha, beta, gamma=0.99, n_actions=4,
                 layer1_size=1024, layer2_size=512, input_dims=8):
        # Discount factor for future rewards
        self.gamma = gamma
        # Learning rate for the actor network
        self.alpha = alpha
        # Learning rate for the critic network
        self.beta = beta
        # Dimensionality of the input observation space
        self.input_dims = input_dims
        # Size of the first hidden layer in the networks
        self.fc1_dims = layer1_size
        # Size of the second hidden layer in the networks
        self.fc2_dims = layer2_size
        # Number of possible actions in the environment
        self.n_actions = n_actions

        # Build and initialize the actor, critic, and policy networks
        self.actor, self.critic, self.policy = self.build_actor_critic_network()
        # List of possible action indices
        self.action_space = [i for i in range(n_actions)]

    # Method to construct the shared actor-critic neural networks
    def build_actor_critic_network(self):
        # Input layer for the state observation
        NN_input = Input(shape=(self.input_dims,))
        # Additional input for the advantage (delta) used in actor loss
        delta = Input(shape=[1])
        # First hidden layer: ReLU activation
        dense1 = Dense(self.fc1_dims, activation='relu')(NN_input)
        # Second hidden layer: ReLU activation
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)
        # Actor output: Softmax probabilities over actions
        probs = Dense(self.n_actions, activation='softmax')(dense2)
        # Critic output: Linear value estimate for the state
        values = Dense(1, activation='linear')(dense2)

        # Custom loss function for the actor: Policy gradient loss weighted by advantage
        def custom_loss(y_true, y_pred):
            # Clip probabilities to avoid log(0)
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            # Log-likelihood of the true action
            log_lik = y_true * K.log(out)
            # Negative log-likelihood scaled by advantage (delta); sum over actions
            return K.sum(-log_lik * delta)

        # Actor model: Takes state and delta, outputs action probabilities
        # Compiled with custom loss and Adam optimizer
        actor = Model(inputs=[NN_input, delta], outputs=[probs])
        actor.compile(optimizer=Adam(lr=self.alpha), loss=custom_loss)

        # Critic model: Takes state, outputs state value
        # Compiled with MSE loss and Adam optimizer
        critic = Model(inputs=[NN_input], outputs=[values])
        critic.compile(optimizer=Adam(lr=self.beta), loss='mean_squared_error')

        # Policy model: Separate model for action selection (shares weights with actor)
        policy = Model(inputs=[NN_input], outputs=[probs])

        return actor, critic, policy

    # Select an action stochastically based on the policy network's probabilities
    def choose_action(self, observation):
        # Reshape observation to batch format (1, input_dims)
        state = observation[np.newaxis, :]
        # Predict action probabilities
        probabilities = self.policy.predict(state)[0]
        # Sample action from categorical distribution
        action = np.random.choice(self.action_space, p=probabilities)
        return action

    # Update the actor and critic networks based on a transition (state, action, reward, next_state, done)
    def learn(self, state, action, reward, state_, done):
        # Reshape states to batch format
        state = state[np.newaxis,:]
        state_ = state_[np.newaxis,:]
        # Predict value of next state (critic)
        critic_value_ = self.critic.predict(state_)
        # Predict value of current state (critic)
        critic_value = self.critic.predict(state)

        # Compute target value: reward + discounted future value (if not done)
        target = reward + self.gamma * critic_value_ * (1 - int(done))
        # Advantage estimate (delta): target - current value
        delta = target - critic_value

        # One-hot encode the taken action for actor input
        actions = np.zeros([1, self.n_actions])
        actions[np.arange(1), action] = 1  # Note: np.arange(1) is [0], so actions[0, action] = 1

        # Update actor: Fit on [state, delta] -> one-hot action
        self.actor.fit([state, delta], actions, verbose=0)

        # Update critic: Fit on state -> target value
        self.critic.fit(state, target, verbose=0) 

# ==============================================================================
# TRAINING FUNCTION
# ==============================================================================
# Main training loop for the A2C agent on LunarLander-v2 environment
def train():
    # Initialize agent with specified learning rates (low alphas for stability)
    agent = Agent(alpha=0.00001, beta=0.00005)

    # Create the LunarLander-v2 environment with human rendering (visual display)
    env = gym.make('LunarLander-v2', render_mode='human')
    # List to track episode scores for averaging
    score_history = []
    # Total number of training episodes
    num_episodes = 2000

    # Loop over episodes
    for i in range(num_episodes):
        # Reset episode flags
        terminated = False
        truncated = False
        score = 0
        # Reset environment and get initial observation
        observation = env.reset()[0]  # Gymnasium returns (obs, info); take obs
        # Run episode until termination or truncation
        while not (terminated or truncated):
            # Render the environment (visualize lander)
            env.render()
            # Select action based on current observation
            action = agent.choose_action(observation)
            # Step environment: get next obs, reward, terminated, truncated, info
            observation_, reward, terminated, truncated, info = env.step(action)
            # Combined done signal
            done = terminated or truncated
            # Learn from this transition
            agent.learn(observation, action, reward, observation_, done)
            # Update observation for next iteration
            observation = observation_
            # Accumulate reward as episode score
            score += reward

        # Record episode score
        score_history.append(score)
        # Compute running average over last 100 episodes (for monitoring convergence)
        avg_score = np.mean(score_history[-100:])
        # Print progress
        print('episode: ', i,'score: %.2f' % score,
              'avg score %.2f' % avg_score)

# Run the training
train()