import os
# Force use of legacy Keras (tf-keras) which is required for keras-rl2
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import gymnasium as gym
import sys
import tensorflow as tf

# ==============================================================================
# COMPATIBILITY PATCHING for keras-rl2 with TensorFlow 2.x and Keras 3
# ==============================================================================

# keras-rl2 is an older library that relies on tf.keras (Keras 2 behavior).
# Modern TensorFlow installs Keras 3 by default, which breaks keras-rl2.
# We use `tf_keras` (a standalone legacy Keras 2 package) and patch it in.

import tf_keras

# Patch 1: Inject __version__ into tf_keras (missing in some versions)
if not hasattr(tf_keras, '__version__'):
    tf_keras.__version__ = '2.15.0'

# Patch 2: Ensure model_from_config is accessible where keras-rl2 expects it
# It looks for it in `keras.models`, but sometimes it's in `keras.saving`.
if not hasattr(tf_keras.models, 'model_from_config'):
     try:
         from tf_keras.saving import model_from_config
         tf_keras.models.model_from_config = model_from_config
     except ImportError:
         pass

# Patch 3: Force `tensorflow.keras` to point to our patched `tf_keras` module.
# This ensures that when keras-rl2 does `from tensorflow.keras import ...`, it gets the legacy version.
tf.keras = tf_keras
sys.modules['tensorflow.keras'] = tf_keras
sys.modules['tensorflow.keras.models'] = tf_keras.models

# ==============================================================================
# IMPORTS (Must use tensorflow.keras / tf_keras)
# ==============================================================================

# Import Keras components from the patched tensorflow.keras (which maps to tf_keras)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
# Use the LEGACY Adam optimizer. The new Keras optimizer causes attribute errors in keras-rl2.
from tensorflow.keras.optimizers.legacy import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

# Wrapper to make the new `gymnasium` library look like the old `gym` library expected by keras-rl2.
class GymnasiumWrapper(gym.Wrapper):
    def step(self, action):
        # Gymnasium returns 5 values: obs, reward, terminated, truncated, info
        # keras-rl2 expects 4 values: obs, reward, done, info
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated or truncated, info
    
    def reset(self, **kwargs):
        # Gymnasium reset returns (obs, info)
        # keras-rl2 expects just obs
        obs, info = self.env.reset(**kwargs)
        return obs
    
    def render(self, mode='human', **kwargs):
        # Keras-rl2 calls render(mode='human')
        # Gymnasium typically handles render_mode at init, so we ignore the argument here.
        return self.env.render()

ENV_NAME = 'CartPole-v0'
# Initialize environment with render_mode='human' to see the window
env = GymnasiumWrapper(gym.make(ENV_NAME, render_mode='human'))

# Set seeds for reproducibility
np.random.seed(123)
# Note: Gymnasium handles seeding differently (via reset), so env.seed(123) is skipped.

nb_actions = env.action_space.n

# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================

# Build a simple Feed-Forward Neural Network for DQN
model = Sequential()
# Flatten input: Observations (4,) -> (1, 4) for temporal difference learning
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
# Output layer: one Q-value per action
model.add(Dense(nb_actions))
model.add(Activation('linear')) 

# ==============================================================================
# AGENT CONFIGURATION
# ==============================================================================

# Experience Replay Memory
memory = SequentialMemory(limit=50000, window_length=1)

# Exploration Policy (Boltzmann/Softmax exploration)
policy = BoltzmannQPolicy()

# Configure DQN Agent
dqn = DQNAgent(
    model=model, 
    nb_actions=nb_actions, 
    memory=memory, 
    nb_steps_warmup=10, 
    target_model_update=1e-2, 
    policy=policy
)

# Compile using the LEGACY Adam optimizer (lr instead of learning_rate)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# ==============================================================================
# TRAINING
# ==============================================================================

# Train the agent
dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)

# Save weights
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Test the agent
dqn.test(env, nb_episodes=5, visualize=True)